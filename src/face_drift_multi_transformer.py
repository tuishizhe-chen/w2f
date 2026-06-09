"""K-slot HIERARCHICAL TRANSFORMER generator for free-phase Words-to-Faces.

Per Yuxuan's slide: instead of K independent conv heads (multi_pixel) or a
single shared theta-MLP (multi_stn), use a two-stage transformer:

  Stage 1 (LAYOUT):   K learnable slot tokens + 1 eps token go through a
                      pre-LN transformer encoder. Slots negotiate WHERE each
                      lives on the 128x128 canvas (theta = r, rot, tx, ty)
                      and emit a coarse semantic "seed vector" describing
                      WHAT lives there. Negotiation = self-attention can see
                      sibling slots, so duplicates are penalised structurally
                      (not just via the overlap loss).

  Stage 2 (RENDER):   Slot tokens are re-injected with stage-1 seed + a
                      positional encoding of stage-1 theta + slot identity,
                      then attend again. Each slot now SEES the finalized
                      layout of its neighbours before painting -> a slot
                      painting an "eye" knows where the other eye sits.
                      Decode: token -> 4x4 spatial seed -> three _up blocks
                      -> 32x32 sigmoid patch.

  Place (STN):        Each patch is placed at its stage-1 theta on the
                      128x128 canvas via inverse-affine grid_sample. The
                      same theta tensor flows BOTH into the affine
                      (STN gradient) AND through theta_pe into stage 2
                      (rendering gradient) so stage 1 receives both signals.

Improvements over the slide design (applied here):
  * theta_head and to_logit final layers zero-init -> at step 0 every slot
    sits at the midpoint radius, centered, zero rotation, with uniform
    0.5 ink. Neutral start matches MultiLayerPixelGen's behaviour.
  * slot_embed orthogonal-init -> aggressive permutation-symmetry break
    for K=12 (Gaussian std 0.02 is too weak with pre-LN).
  * Rotation curriculum: allow_rotation defaults TRUE but is also
    exposed via set_allow_rotation(bool) for the train loop to flip on
    after warm-up if desired.
  * sigmoid_t curriculum: set_sigmoid_t(t) helper updates the value on
    BOTH the base module and the EMA copy from the train loop, so EMA
    samples never lag.

Free-phase: no letter conditioning, only eps. Slot identity is purely
learned via slot_embed. Aligns with the 2026-05 free-phase pivot.
"""
from __future__ import annotations
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import _up
# sweep13's winning face recipe = structural drift kernel (IoU/gradient/...) +
# topk attention. patch_drift_loss_kernel monkey-patches drift_loss._cdist
# globally so BOTH the D2 face drift and the D1 letter drift use the structural
# distance instead of raw-pixel L2 (which mean-face-collapses).
from face_drift_pixel import patch_drift_loss_kernel


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HierarchicalSlotGen(nn.Module):
    """Two-stage hierarchical transformer for free-phase words-to-faces.

    Output: forward(eps[B, d_noise]) -> placed[B, K, 1, 128, 128] in [0,1].
    Same contract as MultiLayerPixelGen -> train loop's composition step
    (sum-then-clip) and per-layer ink-min penalty work unchanged.

    Curriculum hooks (call from train loop):
      set_sigmoid_t(t):       update sigmoid temperature on this module.
      set_allow_rotation(b):  flip rotation on/off without rebuilding.
    For DDP/EMA always call BOTH on the base module AND on the EMA copy.
    """

    def __init__(self,
                 d_noise: int = 128,
                 K: int = 12,
                 d_token: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 patch_size: int = 32,
                 canvas: int = 128,
                 sigmoid_t: float = 1.0,
                 allow_rotation: bool = True,
                 r_min: float = 0.18,
                 r_max: float = 0.55,
                 dxy_max: float = 0.6,
                 letter_mode: bool = False,
                 n_letters: int = 26,
                 alpha_composite: bool = False,
                 slot_intensity: bool = False,
                 letter_film: bool = False):
        super().__init__()
        assert n_layers >= 2 and n_layers % 2 == 0, \
            'n_layers must be even (split equally between stage1 and stage2)'
        self.K = K
        self.d_token = d_token
        self.patch_size = patch_size
        self.canvas = canvas
        self.sigmoid_t = float(sigmoid_t)
        # CHANGE A (alpha compositing): when on, the decoder emits a 2nd
        # (alpha) channel and slots compete for each pixel via softmax-over-K.
        # When off, decoder stays 1-channel and composition is sum().clamp()
        # -> bit-identical to all prior runs. last_placed_alpha is a side
        # channel (see forward) so the return signature is unchanged.
        self.alpha_composite = bool(alpha_composite)
        self.last_placed_alpha = None
        # PER-SLOT UNIFORM INTENSITY (the legit, non-cheating composite). Each
        # slot/layer gets ONE scalar gain (uniform over all pixels), not a
        # per-pixel alpha. canvas = clamp(sum_k gain_k * layer_k). A slot can
        # globally dim/brighten its WHOLE letter, but it CANNOT carve per-pixel
        # -- so it cannot draw the face shape in a free mask independent of the
        # letter content (that per-pixel alpha was cheating: a 2nd free image).
        # Mutually exclusive with alpha_composite; slot_intensity wins if both.
        self.slot_intensity = bool(slot_intensity)
        self.last_slot_logit = None
        self.allow_rotation = bool(allow_rotation)
        self.r_min = float(r_min)
        # Letter-mode: each slot receives an additional letter-class embedding
        # (in addition to the per-slot position embedding and broadcast eps).
        # Together with a per-step letter-image L1 loss this forces each patch
        # to look like a specific letter — gives the model a stroke prior.
        self.letter_mode = bool(letter_mode)
        self.n_letters = int(n_letters)
        # FiLM letter conditioning (2026-06-05, HANDOFF §8 priority 1): instead of
        # ONLY adding letter_embed into the slot tokens (a weak, entangled signal
        # that stage-2 attention smears across all K slots), map letter_embed ->
        # per-CHANNEL (gamma, beta) and modulate each decoder feature map directly:
        # h = gamma * h + beta. This is a multiplicative, per-channel, post-attention
        # conditioning route that the attention cannot dilute. ADDITIVE to the
        # existing token-add + skip-route (kept) so old ckpts/runs stay identical
        # when --letter-film is off. Zero-init the head's last layer -> gamma=1,
        # beta=0 at step 0 (identity modulation), so training starts exactly like a
        # non-FiLM run and learns to deviate.
        self.letter_film = bool(letter_film)
        if self.letter_mode:
            self.letter_embed = nn.Embedding(self.n_letters, d_token)
            # FIX (2026-06-02, H1 conf 0.80): bumped std 0.1 -> 0.5 (5x).
            # Diagnostic on best-run R11 showed class-agnostic blob letters:
            # letter signal entered the seed at ~10% magnitude vs slot_embed
            # (orthogonal ~1.0) and eps_proj (~1.0), then stage-2 attention
            # over K+1=13 tokens mixed it across all slots -> decoder saw a
            # class-agnostic soup. std=0.5 makes the letter signal a real
            # share of the seed without dominating. Pairs with a skip-route
            # injection (self.to_letter_seed) added below so the class signal
            # bypasses stage-2 attention dilution entirely.
            nn.init.normal_(self.letter_embed.weight, std=0.5)
            # SKIP-ROUTE: project letter_embed DIRECTLY into the decoder's
            # 4x4 spatial seed, added AFTER stage-2 attention. Initialized
            # at half scale (weight *= 0.5, bias 0) so initial contribution
            # is moderate, not overwhelming, relative to to_seed(s2_slots).
            self.to_letter_seed = nn.Linear(d_token, 128 * 4 * 4)
            with torch.no_grad():
                self.to_letter_seed.weight.mul_(0.5)
                self.to_letter_seed.bias.zero_()
            # FiLM heads: one per decoder stage we modulate (dec1->96, dec2->64,
            # dec3->32 channels; these three exist for every patch_size). Each maps
            # the per-slot letter embedding to (gamma, beta) per channel. Last
            # layer zero-init so gamma=1+0=1, beta=0 at start (identity).
            if self.letter_film:
                self._film_dims = [96, 64, 32]
                self.film_heads = nn.ModuleList()
                for c in self._film_dims:
                    head = nn.Sequential(
                        nn.LayerNorm(d_token),
                        nn.Linear(d_token, d_token),
                        nn.GELU(),
                        nn.Linear(d_token, 2 * c),
                    )
                    with torch.no_grad():
                        head[-1].weight.zero_()
                        head[-1].bias.zero_()
                    self.film_heads.append(head)
        self.r_max = float(r_max)
        self.dxy_max = float(dxy_max)
        n_per_stage = n_layers // 2

        # Learnable slot identities (orthogonal init -> aggressive symmetry break).
        self.slot_embed = nn.Parameter(torch.empty(K, d_token))
        nn.init.orthogonal_(self.slot_embed)
        # Face-anchor layout prior (STN survey rec#5): K canonical (tx,ty) targets
        # in ~[-0.45,0.45] giving each slot a "job" (eyes/nose/mouth/jaw/...) so
        # slots don't scatter. Used only when --anchor-prior>0; penalty anneals to 0.
        # persistent=False -> NOT saved in state_dict, so old/new ckpts stay compatible.
        if K == 12:
            _anch = [(-0.22, -0.18), (0.22, -0.18), (0.0, -0.05),
                     (-0.18, 0.22), (0.18, 0.22), (-0.42, 0.02), (0.42, 0.02),
                     (-0.34, 0.34), (0.34, 0.34), (0.0, -0.42), (0.0, 0.40), (0.0, 0.12)]
        else:
            _g = max(2, int(math.ceil(math.sqrt(K))))
            _anch = [((i % _g) / (_g - 1) - 0.5, (i // _g) / (_g - 1) - 0.5)
                     for i in range(K)]
        self.register_buffer('anchors',
                             torch.tensor(_anch[:K], dtype=torch.float32),
                             persistent=False)
        # Stage-1 / stage-2 role markers so the transformer knows the phase.
        self.stage1_marker = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.stage2_marker = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        # Eps token role embedding (distinguishes eps from slot tokens).
        self.eps_role = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Eps -> token projection.
        self.eps_proj = nn.Sequential(
            nn.Linear(d_noise, d_token),
            nn.GELU(),
            nn.Linear(d_token, d_token),
        )

        # Stage 1: pre-LN transformer over K+1 tokens.
        enc1 = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
        )
        self.stage1 = nn.TransformerEncoder(enc1, num_layers=n_per_stage)

        # Stage 1 heads.
        #   theta raw -> (r in [r_min,r_max], rot in [-pi/6,pi/6], tx, ty)
        #   seed vec  -> what to render
        self.theta_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Linear(d_token, 4),
        )
        # FIX (2026-06-01): DROPPED zero-init on theta_head and to_logit.
        # Original rationale: "neutral start at midpoint r, centered". But
        # zero-init causes (a) raw_theta == 0 for ALL eps inputs at init ->
        # output is independent of eps, bvar identically 0; (b) backward
        # through zero-W head: dL/dx = W^T @ dL/dy = 0, so stage1/stage2
        # receive ZERO gradient through these paths -- the model can't learn
        # to make outputs depend on eps. Use small-scale random init (default
        # kaiming with the last linear scaled down) so initial raw_theta and
        # patch_logits vary across batch members from step 0.
        with torch.no_grad():
            self.theta_head[-1].weight.mul_(0.1)   # small but non-zero
            self.theta_head[-1].bias.zero_()       # bias 0 -> r centered, rot/tx/ty mean 0

        self.seed_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
        )

        # Positional encoding of stage-1 theta -> stage-2 attention is
        # layout-aware, not just identity-aware (a slot at (0.3,0.4) can
        # attend to its neighbour at (0.35,0.45) and avoid stomping on it).
        # Gradient flows through theta_pe back to raw_theta and stage1, AND
        # also through _affine_grid (which uses the same r,rot,tx,ty
        # tensors). Stage 1 therefore receives BOTH rendering gradient and
        # STN gradient -- desired.
        self.theta_pe = nn.Sequential(
            nn.Linear(4, d_token),
            nn.GELU(),
            nn.Linear(d_token, d_token),
        )

        # Stage 2: pre-LN transformer over K+1 tokens (slots + eps re-used).
        enc2 = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
        )
        self.stage2 = nn.TransformerEncoder(enc2, num_layers=n_per_stage)

        # Stage 2 -> 4x4 spatial seed.
        self.to_seed = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 128 * 4 * 4),
        )

        # Decoder upsample stack: 4x4 seed -> patch_size via log2(patch_size/4)
        # _up blocks. Supports patch_size 32 (3 blocks), 64 (4 blocks).
        # NAMED dec1..dec4 to preserve backward checkpoint compat with the
        # historical patch_size=32 ckpts. dec4 only exists when patch_size>=64.
        assert patch_size in (32, 48, 64), \
            f"patch_size must be 32, 48 or 64, got {patch_size}"
        self.dec1 = _up(128, 96)   # 4 -> 8
        self.dec2 = _up(96, 64)    # 8 -> 16
        self.dec3 = _up(64, 32)    # 16 -> 32
        if patch_size >= 64:
            self.dec4 = _up(32, 32)  # 32 -> 64
        elif patch_size == 48:
            # 32 -> 48 via 1.5x bilinear upsample (done in forward) then
            # refine with Conv3x3+GN+GELU. Kept as a separate layer name so
            # old patch_size=32/64 checkpoints load fine (this attr only
            # exists when patch_size==48).
            self.dec3_to48 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
            )
        # CHANGE A: when alpha_composite, to_logit emits 2 channels
        # (ch0 = content logit, ch1 = alpha logit). Channel 0 keeps the exact
        # historical init (weight*0.1, bias 0) so a checkpoint trained without
        # alpha and one trained with alpha share identical content-channel init.
        out_ch = 2 if self.alpha_composite else 1
        self.to_logit = nn.Conv2d(32, out_ch, 3, 1, 1)
        # FIX (2026-06-01): DROPPED zero-init for the same reason as theta_head
        # above. Bias init linspace(-2, 2) is the MultiLayerPixelGen pattern --
        # without per-slot info available here (slots are interchangeable from
        # to_logit's per-patch view), just keep bias 0 and let weights stay
        # kaiming-default but scale down so initial sigmoid output is near 0.5
        # without being identically 0.5.
        with torch.no_grad():
            self.to_logit.weight.mul_(0.1)
            self.to_logit.bias.zero_()
        if self.alpha_composite:
            # Learnable background alpha logit (scalar). Competes in the
            # softmax-over-(K+bg) so empty regions can stay empty. Init 0 ->
            # at start bg is on equal footing with every slot's raw alpha
            # logit (also ~0 from bias zero-init), i.e. uniform competition.
            self.bg_alpha = nn.Parameter(torch.zeros(()))
        if self.slot_intensity:
            # ONE scalar per slot, read off the stage-2 slot token after layout
            # attention -> gain_k = sigmoid(head(s2_slots_k)) in [0,1], uniform
            # across that slot's pixels. Bias +2 -> initial gain ~sigmoid(2)=0.88
            # ~ near-full, so training starts ~ plain sum().clamp() and the model
            # learns to dim the letters that hurt the face composite.
            self.slot_logit_head = nn.Linear(d_token, 1)
            with torch.no_grad():
                self.slot_logit_head.weight.mul_(0.1)
                self.slot_logit_head.bias.fill_(2.0)

    # --- curriculum hooks ----------------------------------------------------
    def set_sigmoid_t(self, t: float):
        """Update sigmoid temperature. Call on BOTH base and EMA module."""
        self.sigmoid_t = float(t)

    def set_allow_rotation(self, allow: bool):
        """Flip rotation on/off without rebuilding. Call on BOTH base + EMA."""
        self.allow_rotation = bool(allow)

    # --- internals -----------------------------------------------------------
    def _theta_from_raw(self, raw):
        # raw: [B, K, 4]
        r = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw[..., 0])
        if self.allow_rotation:
            rot = (math.pi / 6.0) * torch.tanh(raw[..., 1])
        else:
            rot = torch.zeros_like(raw[..., 1])
        tx = self.dxy_max * torch.tanh(raw[..., 2])
        ty = self.dxy_max * torch.tanh(raw[..., 3])
        return r, rot, tx, ty

    def _affine_grid(self, r, rot, tx, ty, B, K):
        # Inverse affine for grid_sample: canvas (xo,yo) -> patch (xi,yi)
        # so yi = (yo - ty)/r at rot=0. Build per-slot [B,K,2,3], then
        # flatten to [B*K,2,3] (varies K fastest -> matches patch flatten).
        assert r.shape == (B, K), f'r shape {tuple(r.shape)} != ({B},{K})'
        assert rot.shape == (B, K)
        assert tx.shape == (B, K)
        assert ty.shape == (B, K)
        cos = torch.cos(rot)
        sin = torch.sin(rot)
        inv_r = 1.0 / r
        a11 = cos * inv_r
        a12 = sin * inv_r
        a21 = -sin * inv_r
        a22 = cos * inv_r
        b1 = -(a11 * tx + a12 * ty)
        b2 = -(a21 * tx + a22 * ty)
        theta_mat = torch.stack([
            torch.stack([a11, a12, b1], dim=-1),
            torch.stack([a21, a22, b2], dim=-1),
        ], dim=-2)  # [B, K, 2, 3]
        assert theta_mat.shape == (B, K, 2, 3)
        return theta_mat.reshape(B * K, 2, 3)

    def _film(self, h, idx, le_flat):
        """FiLM-modulate decoder feature map h [B*K,C,H,W] with per-slot letter
        embedding le_flat [B*K,D]. gamma=1+head, beta=head -> identity at init."""
        p = self.film_heads[idx](le_flat)              # [B*K, 2C]
        C = h.shape[1]
        gamma = 1.0 + p[:, :C].view(-1, C, 1, 1)
        beta = p[:, C:].view(-1, C, 1, 1)
        return gamma * h + beta

    # --- forward -------------------------------------------------------------
    def forward(self, eps, letter_classes=None, frozen_letter_patches=None):
        """
        eps: [B, d_noise]
        letter_classes: [B, K] long tensor of letter class indices, required
                        when letter_mode=True.
        frozen_letter_patches: [B, K, 1, patch, patch] — if provided AND
                        letter_mode is True, the decoder is SKIPPED and these
                        images are placed directly via STN (sweep24 LetterPlacer
                        pattern). Guarantees patches are literally letter
                        images; model only learns theta.
        Returns:
          letter_mode=False: layers [B, K, 1, 128, 128]
          letter_mode=True:  (layers, patches)
        """
        B = eps.shape[0]
        K = self.K
        D = self.d_token

        # Tokenize.
        # FIX (2026-06-01): broadcast eps to ALL slot tokens, not just a separate
        # eps token. Sweep31 sweep31_K12_d320L6 + K8_d320L6 BOTH showed bvar
        # collapsing toward 0 (step 1500 0.0014 -> step 2400 0.0005) because
        # eps as a single token among K+1=13 gets 1/13 attention weight, drowned
        # out by K constant slot_embed tokens. Result: model ignored eps and
        # converged to a single "average face". Mirroring face_drift_multi_pixel's
        # `fc(eps) -> spatial trunk` pattern: every slot's base now carries eps
        # at full magnitude. Separate eps_tok is kept as an extra context token
        # (cheap, helps stage2 attend over the noise context).
        eps_proj = self.eps_proj(eps)                                    # [B, D]
        eps_tok = (eps_proj + self.eps_role.squeeze(0).squeeze(0)).unsqueeze(1)  # [B,1,D]
        slot_tok = self.slot_embed.unsqueeze(0).expand(B, K, D) \
                   + eps_proj.unsqueeze(1)                               # [B,K,D] ← eps broadcast
        # LETTER MODE: add per-slot letter-class embedding to identify which
        # letter each slot should render. slot_embed still acts as a position
        # token (slot 0 vs slot 1) so "letter A at slot 0" differs from
        # "letter A at slot 1" — the transformer can use position to break
        # ties when the same letter is sampled for multiple slots.
        if self.letter_mode:
            assert letter_classes is not None, 'letter_mode requires letter_classes'
            assert letter_classes.shape == (B, K), f'letter_classes shape {letter_classes.shape} != ({B},{K})'
            slot_tok = slot_tok + self.letter_embed(letter_classes)

        # ---- Stage 1: layout negotiation ----
        s1_in = torch.cat([slot_tok, eps_tok], dim=1) + self.stage1_marker
        s1_out = self.stage1(s1_in)
        s1_slots = s1_out[:, :K]                                          # [B,K,D]
        s1_eps = s1_out[:, K:K + 1]                                       # [B,1,D]

        raw_theta = self.theta_head(s1_slots)                             # [B,K,4]
        r, rot, tx, ty = self._theta_from_raw(raw_theta)
        theta4 = torch.stack([r, rot, tx, ty], dim=-1)                    # [B,K,4]
        # stash realized translations for the optional face-anchor prior (train loop reads it)
        self.last_txy = torch.stack([tx, ty], dim=-1)                     # [B,K,2]

        seed_vec = self.seed_head(s1_slots)                               # [B,K,D]

        # ---- Stage 2: layout-aware rendering ----
        # Re-inject identity + stage1 enrichment + theta-PE + eps (broadcast
        # again so render stage also receives full-magnitude eps signal).
        theta_pe = self.theta_pe(theta4)                                  # [B,K,D]
        s2_slots = seed_vec + theta_pe + self.slot_embed.unsqueeze(0) \
                   + eps_proj.unsqueeze(1)                                # [B,K,D] ← eps broadcast
        if self.letter_mode:
            s2_slots = s2_slots + self.letter_embed(letter_classes)       # letter content cond at render
        s2_in = torch.cat([s2_slots, s1_eps], dim=1) + self.stage2_marker
        s2_out = self.stage2(s2_in)
        s2_slots = s2_out[:, :K]                                          # [B,K,D]
        # PER-SLOT UNIFORM INTENSITY: one scalar logit per slot from its
        # post-layout token. Stashed as a side channel (return signature stays
        # unchanged); train loop / _save_grid read G.last_slot_logit.
        if self.slot_intensity:
            self.last_slot_logit = self.slot_logit_head(s2_slots).squeeze(-1)  # [B,K]
        else:
            self.last_slot_logit = None

        # ---- Decode to 32x32 patch ----
        # In letter_frozen mode the patch is the actual letter image (passed
        # in via frozen_letter_patches); the decoder is skipped entirely.
        # This guarantees patches ARE letters, no blob collapse. Model only
        # learns theta (placement). Sweep24's LetterPlacer pattern.
        # CHANGE A: pre-initialize alpha_patch so the frozen-decoder branch
        # (which never assigns it) does not leave it undefined -> avoids a
        # NameError in the placement guard below. Only set to a real tensor
        # in the alpha_composite decode path.
        alpha_patch = None
        if self.letter_mode and frozen_letter_patches is not None:
            patches = frozen_letter_patches                              # [B, K, 1, 32, 32]
        else:
            # Use .view (not .reshape) so any non-contiguous misalignment
            # fails loudly rather than silently producing a wrong layout.
            seed_flat = self.to_seed(s2_slots)                            # [B,K,128*4*4]
            # SKIP-ROUTE (2026-06-02, H1 fix): inject letter_embed DIRECTLY
            # into the spatial seed AFTER stage-2 attention so the class
            # signal cannot be diluted by 3-layer x 8-head x 13-token mixing.
            # Only active when letter_mode is True (letter_classes provided);
            # non-letter runs remain bit-identical.
            le_flat = None
            if self.letter_mode and letter_classes is not None:
                le = self.letter_embed(letter_classes)                   # [B,K,D]
                seed_flat = seed_flat + self.to_letter_seed(le)          # [B,K,128*4*4]
                if self.letter_film:
                    le_flat = le.reshape(B * K, D)                       # [B*K,D]
            seed = seed_flat.contiguous().view(B * K, 128, 4, 4)
            h = self.dec1(seed)        # 8x8
            if le_flat is not None:
                h = self._film(h, 0, le_flat)
            h = self.dec2(h)           # 16x16
            if le_flat is not None:
                h = self._film(h, 1, le_flat)
            h = self.dec3(h)           # 32x32
            if le_flat is not None:
                h = self._film(h, 2, le_flat)
            if self.patch_size >= 64:
                h = self.dec4(h)       # 64x64
            elif self.patch_size == 48:
                # 32 -> 48 via 1.5x bilinear, then conv refine. Non-pow2
                # path; keeps 32 channels throughout.
                h = F.interpolate(h, size=48, mode='bilinear',
                                  align_corners=False)
                h = self.dec3_to48(h)  # 48x48
            logits = self.to_logit(h)                                    # [B*K,Cout,P,P]
            if self.alpha_composite:
                # ch0 content logit, ch1 raw alpha logit.
                content_logit = logits[:, 0:1]                           # [B*K,1,P,P]
                alpha_logit = logits[:, 1:2]                             # [B*K,1,P,P]
                patches = torch.sigmoid(self.sigmoid_t * content_logit)
                # alpha kept RAW (no sigmoid) -- it goes through softmax-over-K
                # at composition time. View to [B,K,1,P,P] to match `patches`.
                alpha_patch = alpha_logit.view(B, K, 1, self.patch_size,
                                               self.patch_size)
            else:
                patches = torch.sigmoid(self.sigmoid_t * logits)
                alpha_patch = None
            patches = patches.view(B, K, 1, self.patch_size, self.patch_size)

        # ---- STN: place patches on canvas ----
        theta_mat = self._affine_grid(r, rot, tx, ty, B, K)
        grid = F.affine_grid(theta_mat,
                             [B * K, 1, self.canvas, self.canvas],
                             align_corners=False)
        placed = F.grid_sample(
            patches.view(B * K, 1, self.patch_size, self.patch_size),
            grid, mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        out = placed.view(B, K, 1, self.canvas, self.canvas)
        # CHANGE A: place the alpha logit through the SAME grid (same theta) so
        # alpha and content are spatially aligned. padding_mode='zeros' here is
        # acceptable: out-of-patch regions get alpha logit 0, which the
        # composition's softmax(K+bg) lets bg out-compete. Stash on self as a
        # side channel so the return signature stays bit-identical (the train
        # loop / _save_grid read G.last_placed_alpha). Only set when both
        # alpha_composite is on AND the decoder ran (alpha_patch is not None);
        # otherwise None -> callers fall back to the sum().clamp() path.
        if self.alpha_composite and alpha_patch is not None:
            placed_alpha = F.grid_sample(
                alpha_patch.view(B * K, 1, self.patch_size, self.patch_size),
                grid, mode='bilinear', padding_mode='zeros',
                align_corners=False,
            )
            self.last_placed_alpha = placed_alpha.view(
                B, K, 1, self.canvas, self.canvas)
        else:
            self.last_placed_alpha = None
        if self.letter_mode:
            return out, patches
        return out


# ---------------------------------------------------------------------------
# Train loop helpers
# ---------------------------------------------------------------------------
def update_ema(ema, online, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), online.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.bank, weights_only=True)
    N = bank.shape[0]; S_bank = bank.shape[-1]
    assert S_bank == 128
    bank_gpu = (bank.float() / 255.0).to(device)
    real_ink = bank_gpu.mean().item()
    print(f"[multi-trans] bank {tuple(bank.shape)}  real_ink={real_ink:.4f}", flush=True)

    G = HierarchicalSlotGen(
        d_noise=args.d_noise, K=args.K, d_token=args.d_token,
        n_layers=args.n_layers, n_heads=args.n_heads,
        patch_size=args.patch_size, canvas=128,
        sigmoid_t=args.sigmoid_t,
        allow_rotation=(not args.no_rotation),
        r_min=args.r_min, r_max=args.r_max, dxy_max=args.dxy_max,
        letter_mode=bool(args.letter_mode),
        n_letters=args.n_letters,
        alpha_composite=bool(args.alpha_composite),
        slot_intensity=bool(args.slot_intensity),
        letter_film=bool(args.letter_film),
    ).to(device)
    G_ema = copy.deepcopy(G).to(device)
    for p in G_ema.parameters():
        p.requires_grad_(False)
    # ---- optional resume: continue training from a saved checkpoint. The optimizer
    # state and exact curriculum offset are not restored (this is a small project; the
    # LR/curriculum schedule need not be exact), only the weights. Use --start-step so
    # the step counter / sigmoid curriculum pick up roughly where they left off. ----
    if getattr(args, 'init_from', ''):
        _ck = torch.load(args.init_from, map_location=device, weights_only=False)
        _sd_online = _ck.get('G', _ck.get('G_ema', _ck))
        _sd_ema = _ck.get('G_ema', _sd_online)
        G.load_state_dict(_sd_online)
        G_ema.load_state_dict(_sd_ema)
        print(f"[multi-trans] RESUMED weights from {args.init_from} "
              f"(start_step={args.start_step}, target steps={args.steps})", flush=True)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[multi-trans] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}"
          f"  d_token={args.d_token}  n_layers={args.n_layers}  n_heads={args.n_heads}"
          f"  patch={args.patch_size}  letter_mode={args.letter_mode}", flush=True)

    # ---- Letter-mode setup: load EMNIST (or PIL-fallback) letter bank ----
    letter_bank_gpu = None
    if args.letter_mode:
        # IMPORTANT: per-step CPU augmentation (numpy elastic + affine) was
        # bottlenecking training to ~50s/step. Instead, pre-augment a LARGE
        # bank once at startup, cache to disk, and at runtime just sample
        # random precomputed letters from the GPU tensor (microseconds).
        from aug_letters import AugLetterBank, build_aug_bank
        # First load raw EMNIST data (no aug yet)
        raw_bank = AugLetterBank.build(
            args.letter_data_root, size=args.patch_size, device=device,
            level=args.letter_aug_level, extreme_ratio=0.08,
        )
        # Pre-augment: 26 × N_per_class letters, applied once
        cache_path = f'checkpoints/aug_letter_bank_{args.patch_size}_'\
                     f'{args.letter_aug_level}_{args.letter_samples_per_class}.pt'
        aug_bank_cpu = build_aug_bank(
            raw_bank.data, samples_per_class=args.letter_samples_per_class,
            level=args.letter_aug_level, extreme_ratio=0.08,
            cache_path=cache_path,
        )                                                                # [26, N, 1, 32, 32]
        letter_bank_gpu = aug_bank_cpu.to(device)
        print(f"[multi-trans] aug_letter_bank ready: {tuple(letter_bank_gpu.shape)} "
              f"on device", flush=True)
        # fixed letter classes for fixed_eps sampling (so the grid every
        # --sample-every steps draws the SAME letters → fair visual comparison)
        fixed_letter_classes = torch.randint(0, args.n_letters,
                                             (8, args.K), device=device)
        n_per_class = letter_bank_gpu.shape[1]

    # ---- LetterCNN identity loss (2026-06-03) ----
    # Frozen classifier providing deformation-invariant letter legibility.
    # ADDITIVE to D1 drift (which stays — hard requirement), NOT a replacement.
    letter_cnn = None
    if args.letter_mode and args.letter_cls_weight > 0:
        from classifier import load as load_cls
        letter_cnn = load_cls(args.letter_cls_ckpt, device)
        for p in letter_cnn.parameters():
            p.requires_grad_(False)
        letter_cnn.eval()
        print(f"[multi-trans] LetterCNN loaded (frozen) from "
              f"{args.letter_cls_ckpt}, cls_weight={args.letter_cls_weight}",
              flush=True)

    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    # Align the face/letter drift to sweep13's winning recipe: swap raw-pixel L2
    # for a structural kernel (iou/gradient/...). Global — applies to BOTH the D2
    # canvas drift and the D1 per-letter drift.
    # SEPARATE kernels for D2 (face composite) vs D1 (per-letter). IoU is great for
    # FACES but mangles LETTERS into region-fill strokes (loses glyph shape), so we
    # let the face drift use --drift-kernel (e.g. iou) while the letter drift uses
    # --letter-drift-kernel (e.g. l2) to preserve letter form. The global _cdist is
    # re-patched right before each drift call (cheap function-pointer swap).
    # User 2026-06-05: "弃IoU" meant drop IoU ON LETTERS, not on faces.
    print(f"[multi-trans] face(D2) kernel={args.drift_kernel} "
          f"(topk {args.face_topk_pos}/{args.face_topk_neg}) | "
          f"letter(D1) kernel={args.letter_drift_kernel}", flush=True)
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    t0 = time.time()

    # precompute coord grids for locality 2nd-moment penalty
    if args.locality > 0:
        ys_g = torch.linspace(0., 1., 128, device=device).view(1, 1, 128, 1)
        xs_g = torch.linspace(0., 1., 128, device=device).view(1, 1, 1, 128)
    G.train()
    for step in range(args.start_step + 1, args.steps + 1):
        prog = step / max(1, args.steps)
        # sigmoid_t curriculum -- mutate on BOTH base and EMA copy.
        if args.sigmoid_t_end is not None:
            cur_sig_t = args.sigmoid_t + (args.sigmoid_t_end - args.sigmoid_t) * prog
            G.set_sigmoid_t(cur_sig_t); G_ema.set_sigmoid_t(cur_sig_t)
        sharp_w = args.sharpness if args.sharpness_end is None else \
                  args.sharpness + (args.sharpness_end - args.sharpness) * prog

        eps = torch.randn(args.bgen, args.d_noise, device=device)
        if args.letter_mode:
            letter_classes = torch.randint(0, args.n_letters,
                                           (args.bgen, args.K), device=device)
            # In frozen mode, sample letter patches NOW and pass to model so
            # decoder is bypassed.
            if args.letter_frozen:
                rand_idx = torch.randint(0, n_per_class,
                                         (args.bgen, args.K), device=device)
                frozen_patches = letter_bank_gpu[letter_classes, rand_idx]
                layers, patches = G(eps, letter_classes,
                                    frozen_letter_patches=frozen_patches)
            else:
                layers, patches = G(eps, letter_classes)
        else:
            letter_classes = None
            patches = None
            layers = G(eps)                                              # [B, K, 1, 128, 128]
        # CHANGE A: softmax-over-slots alpha compositing. G stashes the placed
        # alpha logits on G.last_placed_alpha [B,K,1,128,128]. Stack a learnable
        # background alpha logit (broadcast to [B,1,1,128,128]) as an extra
        # competitor, softmax over the K+1 dim, then canvas = sum_k weight_k *
        # content_k. bg lets empty regions stay empty. When the flag is off (or
        # alpha unavailable, e.g. frozen decoder), fall back to sum().clamp() ->
        # bit-identical to prior runs.
        slot_logit = getattr(G, 'last_slot_logit', None)
        placed_alpha = getattr(G, 'last_placed_alpha', None)
        if args.slot_intensity and slot_logit is not None:
            # LEGIT composite: per-slot UNIFORM gain in [0,1], then sum().clamp().
            # gain_k scales the WHOLE layer k (same value every pixel) -> a slot
            # cannot carve per-pixel; it can only dim/brighten its entire letter.
            Bc = layers.shape[0]
            gain = torch.sigmoid(slot_logit).view(Bc, G.K, 1, 1, 1)      # [B,K,1,1,1]
            weighted = gain * layers                                     # [B,K,1,128,128]
            sum_layers = weighted.sum(dim=1)                             # [B,1,128,128]
            canvas = sum_layers.clamp(0.0, 1.0)
            sum_sq = sum_layers.pow(2)
            sq_sum = weighted.pow(2).sum(dim=1)
            overlap_val = ((sum_sq - sq_sum) / 2.0).mean()
        elif args.alpha_composite and placed_alpha is not None:
            Bc = layers.shape[0]
            bg = G.bg_alpha.view(1, 1, 1, 1, 1).expand(Bc, 1, 1, 128, 128)
            alpha_all = torch.cat([placed_alpha, bg], dim=1)             # [B,K+1,1,128,128]
            w = torch.softmax(alpha_all, dim=1)                          # over K+1 slots
            w_slots = w[:, :G.K]                                         # [B,K,1,128,128]
            canvas = (w_slots * layers).sum(dim=1).clamp(0.0, 1.0)       # [B,1,128,128]
            # overlap on the alpha-weighted layers (competition already
            # suppresses double-counting, but keep the same penalty form).
            weighted = w_slots * layers
            sum_layers = weighted.sum(dim=1)
            sum_sq = sum_layers.pow(2)
            sq_sum = weighted.pow(2).sum(dim=1)
            overlap_val = ((sum_sq - sq_sum) / 2.0).mean()
        else:
            sum_layers = layers.sum(dim=1)                               # [B, 1, 128, 128]
            canvas = sum_layers.clamp(0.0, 1.0)
            # Pairwise overlap: sum of (layer_i * layer_j) for i<j == (sum^2 - Sigma layer^2)/2
            sum_sq = sum_layers.pow(2)
            sq_sum = layers.pow(2).sum(dim=1)
            overlap_val = ((sum_sq - sq_sum) / 2.0).mean()

        # drift on the COMPOSITE canvas -- pixel-space, same recipe as multi_pixel.
        canvas_feat = canvas.flatten(1)
        Cg = max(1, args.gen_per)
        B = args.bgen // Cg
        gen_arg = canvas_feat[: B * Cg].view(B, Cg, -1)
        pos_idx = torch.randint(0, N, (B, args.cp))
        pos_feat = bank_gpu[pos_idx].flatten(2)
        neg_idx = torch.randint(0, N, (B, args.cn))
        neg_feat = bank_gpu[neg_idx].flatten(2)
        if args.noise_aug > 0:
            t = args.noise_aug
            gen_arg = (1 - t) * gen_arg + t * torch.randn_like(gen_arg)
            pos_feat = (1 - t) * pos_feat + t * torch.randn_like(pos_feat)
            neg_feat = (1 - t) * neg_feat + t * torch.randn_like(neg_feat)
        # D2 = face composite drift -> use the FACE kernel (e.g. iou).
        patch_drift_loss_kernel(args.drift_kernel)
        loss_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat, R_list=R_list,
                                    topk_pos=args.face_topk_pos,
                                    topk_neg=args.face_topk_neg)
        loss = loss_vec.mean()
        # loss-component magnitudes (weighted contribution of each term) for the
        # adaptive-weight study. Stored as detached scalars; .item() only at log step.
        lcomp = {'d2_face': loss.detach()}

        if args.cov > 0:
            cov_t = args.cov * (canvas.mean() - real_ink).pow(2)
            lcomp['cov'] = cov_t.detach(); loss = loss + cov_t
        if sharp_w > 0:
            sharp_t = sharp_w * (1.0 - 4.0 * (canvas - 0.5).pow(2)).mean()
            lcomp['sharp'] = sharp_t.detach(); loss = loss + sharp_t
        if args.tv > 0:
            dx = (canvas[:, :, :, 1:] - canvas[:, :, :, :-1]).abs().mean()
            dy = (canvas[:, :, 1:, :] - canvas[:, :, :-1, :]).abs().mean()
            loss = loss + args.tv * (dx + dy)
        if args.overlap > 0:
            ov_t = args.overlap * overlap_val
            lcomp['overlap'] = ov_t.detach(); loss = loss + ov_t
        # Region-IoU (sweep24's winning trick for letter composition): force the
        # canvas to have ink IN THE SAME REGIONS as a sampled real face. Global
        # drift only constrains canvas-level statistics so the model plateaus at
        # "scattered patches with correct stats but wrong placement". Per-region
        # IoU is the lever that pushes patches into face-anatomical positions.
        if args.region_iou > 0:
            ps = 128 // args.n_regions
            ridx2 = torch.randint(0, N, (canvas.shape[0],))
            tgt = bank_gpu[ridx2]                                   # [B, 128, 128]
            cR = F.avg_pool2d(canvas[:, 0].unsqueeze(1), ps).squeeze(1)
            tR = F.avg_pool2d(tgt.unsqueeze(1), ps).squeeze(1)
            inter = (cR * tR).sum(dim=(-1, -2))
            union = cR.sum(dim=(-1, -2)) + tR.sum(dim=(-1, -2)) - inter
            riou_t = args.region_iou * (1.0 - inter / (union + 1e-6)).mean()
            lcomp['region_iou'] = riou_t.detach(); loss = loss + riou_t
        if args.region_cov > 0:
            ps = 128 // args.n_regions
            ridx_rc = torch.randint(0, N, (canvas.shape[0],))
            canvas_reg = F.avg_pool2d(canvas, ps)
            real_reg = F.avg_pool2d(bank_gpu[ridx_rc].unsqueeze(1), ps)
            loss = loss + args.region_cov * F.mse_loss(canvas_reg, real_reg)
        # LETTER L1 loss (letter_mode only): force each decoded patch to look
        # like its assigned letter. AugLetterBank.sample applies augmentation
        # (elastic + affine) per-call so two slots with the same letter class
        # get slightly different targets — model learns "letter A variants",
        # not a single canonical glyph. This is the stroke-prior the user
        # asked for: letters have line structure, blocks don't.
        # Letter weight curriculum (if --letter-weight-end is set, linearly
        # anneal from letter_weight to letter_weight_end over training).
        cur_letter_w = args.letter_weight if args.letter_weight_end is None else \
                       args.letter_weight + (args.letter_weight_end - args.letter_weight) * prog
        # ------------------------------------------------------------------
        # LETTER LOSS BLOCK (mode-selectable, 2026-06-02)
        # Per W2F proposal D1:
        #   L_D1 = Σ_i || y_i - sg(y_i + V_p^{x_i}(y_i) - V_q^{x_i}(y_i)) ||²
        # V_p draws from REAL same-class letters (letter_bank_gpu[c]);
        # V_q draws from OTHER same-class GENERATED patches in the batch.
        # Mode 'l1': original L1 (vCurr default, bit-identical).
        # Mode 'drift': proposal D1 per-class drift (Cg=1, Cp=K_pos real,
        #               Cn=K_neg same-class generated, self-excluded).
        # Mode 'both': L1 (0.5x) + drift (0.5x) of the curriculum weight,
        #              then drift sub-scaled by letter_drift_weight (audit:
        #              drift returns scaled MSE O(1) vs L1 O(0.1); without
        #              sub-weight curriculum 5→0.3 calibrated for L1 makes
        #              drift effectively 5-30× too strong).
        # ------------------------------------------------------------------
        d1_skipped_classes = 0
        d1_per_R_info = {}
        # Keep the letter terms SEPARATE (D1 drift, CE, optional L1) so each can be
        # adapted INDEPENDENTLY to its own target ratio of the face pressure (--letter-adapt).
        d1_term = None
        ce_t = None
        l1_term = None
        if args.letter_mode and cur_letter_w > 0 and not args.letter_frozen:
            mode = args.letter_loss_mode
            if mode in ('l1', 'both'):
                rand_idx = torch.randint(0, n_per_class,
                                         (args.bgen, args.K), device=device)
                letter_tgt = letter_bank_gpu[letter_classes, rand_idx]
                l1_scale = 1.0 if mode == 'l1' else 0.5
                l1_term = l1_scale * cur_letter_w * F.l1_loss(patches, letter_tgt)
            if mode in ('drift', 'both'):
                D_p = args.patch_size * args.patch_size
                flat_patches = patches.reshape(args.bgen * args.K, D_p)
                flat_classes = letter_classes.reshape(-1)
                Cp_d1 = int(args.letter_drift_cp)
                Cn_d1 = int(args.letter_drift_cn)
                R_list_let = tuple(float(r) for r in args.letter_drift_R.split(','))
                t_noise_let = float(args.letter_noise_aug)
                # bug A fix: separate sub-weight for drift to compensate magnitude diff
                drift_subw = float(args.letter_drift_weight)
                outer_scale = 1.0 if mode == 'drift' else 0.5
                # D1 = per-letter drift -> use the LETTER kernel (e.g. l2) so glyph
                # shape is preserved (IoU would fill regions and destroy letters).
                patch_drift_loss_kernel(args.letter_drift_kernel)
                total_drift = flat_patches.new_zeros(())
                n_total = 0
                for c in range(args.n_letters):
                    mask_c = (flat_classes == c).nonzero(as_tuple=False).squeeze(-1)
                    n_c = int(mask_c.numel())
                    min_needed = 2 if Cn_d1 > 0 else 1
                    if n_c < min_needed:
                        d1_skipped_classes += 1
                        continue
                    gen_c = flat_patches[mask_c].unsqueeze(1)
                    pos_idx = torch.randint(0, n_per_class, (n_c, Cp_d1), device=device)
                    pos_c = letter_bank_gpu[c, pos_idx].reshape(n_c, Cp_d1, D_p)
                    if Cn_d1 > 0:
                        if n_c - 1 >= Cn_d1:
                            rp = torch.rand(n_c, n_c, device=device)
                            rp.scatter_(1, torch.arange(n_c, device=device).unsqueeze(1), -1.0)
                            neg_local = rp.topk(Cn_d1, dim=1).indices
                        else:
                            neg_local = torch.randint(0, n_c, (n_c, Cn_d1), device=device)
                            sh = neg_local == torch.arange(n_c, device=device).unsqueeze(1)
                            neg_local = torch.where(sh, (neg_local + 1) % n_c, neg_local)
                        neg_c = flat_patches[mask_c[neg_local]].detach()
                    else:
                        neg_c = flat_patches.new_zeros((n_c, 0, D_p))
                    if t_noise_let > 0:
                        gen_c = (1 - t_noise_let) * gen_c + t_noise_let * torch.randn_like(gen_c)
                        pos_c = (1 - t_noise_let) * pos_c + t_noise_let * torch.randn_like(pos_c)
                        if neg_c.shape[1] > 0:
                            neg_c = (1 - t_noise_let) * neg_c + t_noise_let * torch.randn_like(neg_c)
                    loss_vec_c, info_c = drift_loss(
                        gen_c, pos_c,
                        fixed_neg=neg_c if neg_c.shape[1] > 0 else None,
                        R_list=R_list_let,
                    )
                    total_drift = total_drift + loss_vec_c.sum()
                    n_total += n_c
                    for k_info, v_info in info_c.items():
                        if k_info.startswith('loss_R'):
                            d1_per_R_info[k_info] = d1_per_R_info.get(k_info, 0.0) + float(v_info) * n_c
                if n_total > 0:
                    d1_loss = total_drift / n_total
                    d1_term = outer_scale * cur_letter_w * drift_subw * d1_loss
                    lcomp['d1_letter'] = d1_term.detach()
                    for k_info in d1_per_R_info:
                        d1_per_R_info[k_info] /= n_total
        # LetterCNN identity loss (2026-06-03): deformation-invariant legibility.
        # Adds a "this patch must classify as letter c" pressure that tolerates
        # face-conforming deformation (a Z stretched along a jaw still reads Z),
        # so D1 drift (kept above) can run at a lower weight and free patch
        # plasticity for D2 face composition. Frozen CNN -> grad flows only to
        # the decoder, never to the classifier.
        if (letter_cnn is not None and args.letter_cls_weight > 0
                and not args.letter_frozen):
            cls_in = patches.reshape(args.bgen * args.K, 1,
                                     args.patch_size, args.patch_size)
            if args.patch_size != 32:
                cls_in = F.interpolate(cls_in, size=32, mode='bilinear',
                                       align_corners=False)
            cls_logits = letter_cnn(cls_in)                              # [B*K, 26]
            cls_loss = F.cross_entropy(cls_logits, letter_classes.reshape(-1))
            ce_t = args.letter_cls_weight * cls_loss
            lcomp['ce_letter'] = ce_t.detach()

        # ---- adaptive letter weights: D1-drift and CE adapted SEPARATELY ----------
        # Standard per-step loss-balancing, NO EMA: each term is divided by its OWN
        # current magnitude and scaled to ratio_i x the current face pressure
        # F = D2_face + region-IoU. So D1's value is pinned to ratio_d1*F and CE's to
        # ratio_ce*F every step (grad direction preserved, magnitude normalized). This
        # keeps BOTH letter signals alive independently (CE otherwise collapses to ~0).
        wa_d1 = 1.0; wa_ce = 1.0
        face_cur = lcomp['d2_face'].detach()
        if 'region_iou' in lcomp:
            face_cur = face_cur + lcomp['region_iou'].detach()
        for _term, _ratio, _wname in ((l1_term, args.letter_adapt_ratio, 'l1'),
                                      (d1_term, args.letter_adapt_ratio, 'd1'),
                                      (ce_t, args.ce_adapt_ratio, 'ce')):
            if _term is None:
                continue
            if args.letter_adapt and float(_term.detach()) > 1e-9:
                _w = (_ratio * face_cur / (_term.detach() + 1e-6)).clamp(0.02, 100.0)
                if _wname == 'd1':
                    wa_d1 = _w
                elif _wname == 'ce':
                    wa_ce = _w
                loss = loss + _w * _term
            else:
                loss = loss + _term
        # ---- per-patch denoising (TV + sharpness on the 32x32 glyphs themselves) ----
        # Targets the "noisy letter layers" directly: the structural face drift
        # backprops texture into the patch CONTENT; a TV / sharpness applied to the
        # patches (not just the composite canvas) smooths each glyph without touching
        # its placement. Off by default.
        if patches is not None and (args.patch_tv > 0 or args.patch_sharp > 0):
            if args.patch_tv > 0:
                pdx = (patches[..., 1:] - patches[..., :-1]).abs().mean()
                pdy = (patches[..., 1:, :] - patches[..., :-1, :]).abs().mean()
                loss = loss + args.patch_tv * (pdx + pdy)
            if args.patch_sharp > 0:
                loss = loss + args.patch_sharp * (1.0 - 4.0 * (patches - 0.5).pow(2)).mean()
        if args.locality > 0:
            w = layers.squeeze(2)                                        # [B, K, 128, 128]
            mass = w.sum(dim=(2, 3)).clamp_min(1e-3)                     # [B, K]
            cy = (w * ys_g).sum(dim=(2, 3)) / mass                       # [B, K]
            cx = (w * xs_g).sum(dim=(2, 3)) / mass
            var = (w * ((ys_g - cy[..., None, None]).pow(2)
                       + (xs_g - cx[..., None, None]).pow(2))).sum(dim=(2, 3)) / mass
            loss = loss + args.locality * var.mean()
        # Face-anchor layout prior (STN survey rec#5): pull each slot's realized
        # (tx,ty) toward its canonical anchor, weight annealed to 0 over training
        # (guides early layout so slots tile face parts, then releases). default off.
        if args.anchor_prior > 0 and getattr(G, 'last_txy', None) is not None:
            anchor_w = args.anchor_prior * (1.0 - prog)
            loss = loss + anchor_w * (G.last_txy - G.anchors).pow(2).mean()
        # anti-collapse: primary defense against duplicate-slot collapse since
        # this model uses standard transformer attention (no mutual-exclusion
        # by construction). User memory recommends layer_min_w ~100-200.
        if args.layer_min > 0:
            per_layer_ink = layers.mean(dim=(0, 2, 3, 4))                # [K]
            dead = torch.relu(args.layer_min - per_layer_ink).sum()
            loss = loss + args.layer_min_w * dead

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        if args.ema > 0:
            update_ema(G_ema, G, args.ema)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = canvas.var(dim=0).mean().item()
                ink = canvas.mean().item()
                per_layer_ink = layers.mean(dim=(0, 2, 3, 4)).cpu().numpy()
            alive = sum(1 for v in per_layer_ink if v > 0.005)
            _lc = ' '.join(f"{k}={v.item():.3f}" for k, v in lcomp.items())
            _wa = f" wa_d1={float(wa_d1):.3f} wa_ce={float(wa_ce):.3f}" if args.letter_adapt else ""
            print(f"step={step} loss={loss.item():.3f} ink={ink:.3f} bvar={bvar:.4f} "
                  f"alive={alive}/{args.K} ovl={overlap_val.item():.4f} "
                  f"grad={gnorm.item():.2f} t={time.time()-t0:.0f}s | LC {_lc}{_wa}", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            G_for_vis = G_ema if args.ema > 0 else G
            # For viz in frozen mode, sample fixed letter patches once and use
            # them every viz call (deterministic).
            fixed_patches = None
            if args.letter_mode and args.letter_frozen:
                # pick a stable random idx per (b,k) via class-indexed lookup
                # at the median letter slot
                f_idx = torch.randint(0, n_per_class,
                                      fixed_letter_classes.shape, device=device,
                                      generator=torch.Generator(device=device).manual_seed(0))
                fixed_patches = letter_bank_gpu[fixed_letter_classes, f_idx]
            _save_grid(G_for_vis, fixed_eps, bank_gpu,
                       out / f'multitrans_step{step:05d}.png', device, args.K,
                       fixed_letter_classes if args.letter_mode else None,
                       fixed_patches)

        # Intermediate checkpoint so an interrupted run isn't a total loss.
        # Atomic rename via a .tmp file -> if shutdown mid-save the old ckpt
        # stays intact. Keeps a single rolling ckpt to bound disk use.
        if args.ckpt_every > 0 and (step % args.ckpt_every == 0 or step == args.steps):
            ckpt_path = out / 'G_ckpt.pt'
            tmp_path = out / 'G_ckpt.pt.tmp'
            torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(),
                        'opt': opt.state_dict(), 'step': step, 'args': vars(args)},
                       tmp_path)
            tmp_path.replace(ckpt_path)
            print(f"[multi-trans] ckpt @ step {step} -> {ckpt_path}", flush=True)
            # KEEP-CKPTS (2026-06-05): the rolling single G_ckpt.pt cost us the
            # step-6000 peak (sweep44 KID 79 -> 206 by step16000; the model peaks
            # EARLY then over-trains). When --keep-ckpts is set, also persist a
            # per-step copy so we can eval each and lock the true best. Data disk
            # is ample (~75MB/ckpt). Default off = unchanged rolling behaviour.
            if args.keep_ckpts:
                step_path = out / f'G_step{step:05d}.pt'
                torch.save({'G_ema': G_ema.state_dict(), 'step': step,
                            'args': vars(args)}, step_path)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'args': vars(args)},
               out / 'G_final.pt')
    print(f"[multi-trans] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, fixed_eps, bank, path, device, K,
               fixed_letter_classes=None, fixed_letter_patches=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    patches_t = None
    if fixed_letter_classes is not None:
        out = G(fixed_eps, fixed_letter_classes,
                frozen_letter_patches=fixed_letter_patches)
        layers_t, patches_t = out[0], out[1]   # patches_t [N,K,1,P,P] raw per-slot content
    else:
        layers_t = G(fixed_eps)                                              # [N, K, 1, 128, 128]
    # CHANGE A: viz must use the SAME composition as training so the grid
    # reflects what the loss sees. If alpha was placed, softmax-over-(K+bg)
    # then weighted sum; else sum().clamp(). Done on the GPU tensor before
    # converting to numpy.
    slot_logit = getattr(G, 'last_slot_logit', None)
    placed_alpha = getattr(G, 'last_placed_alpha', None)
    if getattr(G, 'slot_intensity', False) and slot_logit is not None:
        Bc = layers_t.shape[0]
        gain = torch.sigmoid(slot_logit).view(Bc, G.K, 1, 1, 1)
        canvas_t = (gain * layers_t).sum(dim=1).clamp(0.0, 1.0)              # [N,1,128,128]
        canvas = canvas_t[:, 0].cpu().numpy()                               # [N,128,128]
    elif getattr(G, 'alpha_composite', False) and placed_alpha is not None:
        Bc = layers_t.shape[0]
        bg = G.bg_alpha.view(1, 1, 1, 1, 1).expand(Bc, 1, 1, 128, 128)
        alpha_all = torch.cat([placed_alpha, bg], dim=1)
        w = torch.softmax(alpha_all, dim=1)[:, :G.K]
        canvas_t = (w * layers_t).sum(dim=1).clamp(0.0, 1.0)                 # [N,1,128,128]
        canvas = canvas_t[:, 0].cpu().numpy()                               # [N,128,128]
    else:
        canvas = np.clip(layers_t.sum(dim=1)[:, 0].cpu().numpy(), 0.0, 1.0)
    layers = layers_t.cpu().numpy()
    G.train()
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)
    ridx = torch.randint(0, bank.shape[0], (fixed_eps.shape[0],))
    real = bank[ridx].cpu().numpy()
    Nf = fixed_eps.shape[0]
    fig, axes = plt.subplots(3, Nf, figsize=(Nf * 1.8, 6), facecolor='#0d0f14')
    for i in range(Nf):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += layers[i, k, 0][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
        # label which 12 letters this sample is composed of (above the layer row)
        if fixed_letter_classes is not None:
            _abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            _lstr = ''.join(_abc[int(c)] for c in fixed_letter_classes[i].cpu().numpy())
            axes[0, i].set_title(_lstr, color='#9cf', fontsize=6,
                                 family='monospace', pad=2)
    fig.suptitle(f'Multi-layer transformer K={K} -- {path.stem}  '
                 '(top=layers, mid=MODEL output, bottom=real dataset)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.01,
                        hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[multi-trans] grid -> {path}", flush=True)

    # Per-slot letter breakdown: row = one sample, col = one of the K slots,
    # cell = that slot's decoded 32x32 patch titled with its TARGET letter. Lets
    # you verify slot k actually drew the letter it was assigned (the main grid
    # only shows the merged composite, so you can't tell per-letter quality).
    if fixed_letter_classes is not None and patches_t is not None:
        _abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        pnp = patches_t[:, :, 0].cpu().numpy()                 # [N,K,P,P]
        lc = fixed_letter_classes.cpu().numpy()                # [N,K]
        figL, axesL = plt.subplots(Nf, K, figsize=(K * 0.85, Nf * 1.0),
                                   facecolor='#0d0f14')
        axesL = np.atleast_2d(axesL)
        for i in range(Nf):
            for k in range(K):
                axesL[i, k].imshow(pnp[i, k], cmap='gray', vmin=0, vmax=1)
                axesL[i, k].axis('off')
                axesL[i, k].set_title(_abc[int(lc[i, k])], color='#9cf',
                                      fontsize=8, pad=1)
        figL.suptitle(f'Per-slot decoded patches (title = target letter) K={K} '
                      f'-- {path.stem}', color='white', fontsize=11)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01,
                            hspace=0.3, wspace=0.05)
        lpath = path.with_name(path.stem + '_letters' + path.suffix)
        plt.savefig(lpath, dpi=100, facecolor='#0d0f14'); plt.close(figL)
        print(f"[multi-trans] letter grid -> {lpath}", flush=True)

    # Per-layer BACKUP (user request 2026-06-05): every time we save a viz, also
    # dump each slot's PLACED 128x128 layer so we can debug what every layer
    # contributed without re-running / without pulling everything. Two artifacts:
    #   *_layers.npy : compact uint8 [N,K,128,128], faithful (reconstruct any layer)
    #   *_layers.png : viewable grid, row=sample, col=slot (titled with its letter)
    layers_u8 = (np.clip(layers[:, :, 0], 0.0, 1.0) * 255).astype(np.uint8)  # [N,K,128,128]
    np.save(path.with_name(path.stem + '_layers.npy'), layers_u8)
    figP, axesP = plt.subplots(Nf, K, figsize=(K * 0.85, Nf * 0.95),
                               facecolor='#0d0f14')
    axesP = np.atleast_2d(axesP)
    _abc2 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(Nf):
        for k in range(K):
            axesP[i, k].imshow(layers_u8[i, k], cmap='gray', vmin=0, vmax=255)
            axesP[i, k].axis('off')
            if fixed_letter_classes is not None:
                axesP[i, k].set_title(_abc2[int(fixed_letter_classes[i, k])],
                                      color='#9cf', fontsize=7, pad=1)
    figP.suptitle(f'Per-slot PLACED layers on 128 canvas K={K} -- {path.stem}',
                  color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01,
                        hspace=0.3, wspace=0.05)
    ppath = path.with_name(path.stem + '_layers' + path.suffix)
    plt.savefig(ppath, dpi=100, facecolor='#0d0f14'); plt.close(figP)
    print(f"[multi-trans] per-layer backup -> {ppath} (+ .npy)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor-prior', type=float, default=0.0, dest='anchor_prior',
                    help='Face-anchor layout prior weight (STN survey rec#5). Pulls each '
                         'slot (tx,ty) toward a canonical face-part anchor, annealed to 0 '
                         'over training. 0 = off (default, bit-identical to prior runs).')
    ap.add_argument('--bank', required=True)
    ap.add_argument('--out', default='./samples/multi_transformer')
    ap.add_argument('--K', type=int, default=12)
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=256)
    ap.add_argument('--base', type=int, default=192,
                    help='kept for CLI compatibility with multi_pixel.py; '
                         'transformer generator does not use it.')
    ap.add_argument('--cp', type=int, default=64)
    ap.add_argument('--cn', type=int, default=16)
    ap.add_argument('--gen-per', type=int, default=32, dest='gen_per')
    ap.add_argument('--steps', type=int, default=24000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.005,0.02,0.1')
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--cov', type=float, default=5.0)
    ap.add_argument('--tv', type=float, default=0.03)
    ap.add_argument('--init-from', type=str, default='', dest='init_from',
                    help='resume: load G/G_ema weights from this checkpoint before training.')
    ap.add_argument('--start-step', type=int, default=0, dest='start_step',
                    help='resume: begin the step counter here (continues the curriculum).')
    ap.add_argument('--sharpness', type=float, default=0.5)
    ap.add_argument('--sharpness-end', type=float, default=4.0, dest='sharpness_end')
    ap.add_argument('--sigmoid-t', type=float, default=1.0, dest='sigmoid_t')
    ap.add_argument('--sigmoid-t-end', type=float, default=3.0, dest='sigmoid_t_end')
    ap.add_argument('--noise-aug', type=float, default=0.10, dest='noise_aug')
    ap.add_argument('--overlap', type=float, default=5.0,
                    help='pairwise overlap penalty weight Sigma_{i<j} (layer_i * layer_j). '
                         'User priority: face > letter > no-overlap. Keep LOW.')
    ap.add_argument('--locality', type=float, default=0.0,
                    help='per-layer 2nd-moment locality penalty weight; in normalized '
                         '[0,1]^2 coords so typical scale 5e-4 to 5e-3.')
    ap.add_argument('--layer-min', type=float, default=0.005, dest='layer_min',
                    help='min mean ink per layer; layers below this get penalized '
                         '(anti-collapse). 0 disables.')
    ap.add_argument('--layer-min-w', type=float, default=150.0, dest='layer_min_w',
                    help='weight on the layer-min anti-collapse penalty (transformer has '
                         'no built-in mutual exclusion -> bump to ~150 vs pixel default 100).')
    ap.add_argument('--region-iou', type=float, default=0.0, dest='region_iou',
                    help='per-region IoU weight: divide canvas into n_regions x n_regions, '
                         'compute IoU vs a sampled real face per region, average. The '
                         'sweep24 winning trick — forces face-anatomical patch placement.')
    ap.add_argument('--region-cov', type=float, default=0.0, dest='region_cov',
                    help='per-region cov weight: MSE between canvas per-region mean ink '
                         'and a sampled real face per-region mean ink. Softer than region-IoU.')
    ap.add_argument('--n-regions', type=int, default=8, dest='n_regions',
                    help='spatial regions per side for region-IoU/cov (8 = 16x16 px regions).')
    # Letter mode (2026-06-01): condition each slot on a letter class + add an
    # L1 letter-content loss so patches look like letters, not blocks. Gives the
    # model a stroke prior the free-form patches lacked.
    ap.add_argument('--letter-mode', action='store_true', dest='letter_mode',
                    help='enable letter conditioning: each slot gets a sampled letter class, '
                         'patch must match the corresponding letter image (L1 loss).')
    ap.add_argument('--n-letters', type=int, default=26, dest='n_letters')
    ap.add_argument('--letter-film', action='store_true', dest='letter_film',
                    help='FiLM letter conditioning (HANDOFF §8 #1): map per-slot letter '
                         'embedding -> per-channel (gamma,beta) and modulate each decoder '
                         'feature map (h=gamma*h+beta). A strong, per-channel, '
                         'post-attention route the stage-2 attention cannot dilute. '
                         'ADDITIVE to token-add + skip-route (both kept). Zero-init -> '
                         'identity at step 0, so default OFF is bit-identical to prior runs.')
    ap.add_argument('--letter-weight', type=float, default=1.0, dest='letter_weight',
                    help='weight on the L1 letter-content loss (canvas losses unchanged).')
    ap.add_argument('--letter-weight-end', type=float, default=None, dest='letter_weight_end',
                    help='if set, linearly anneal letter_weight from --letter-weight to this value '
                         'over training. Useful for curriculum: high early (force letter strokes) '
                         'then decay (let drift+region-IoU sculpt placement). Mirrors sharpness curriculum.')
    ap.add_argument('--letter-data-root', default='./data/emnist', dest='letter_data_root',
                    help='EMNIST download path (torchvision); falls back to PIL letters if download fails.')
    ap.add_argument('--letter-aug-level', default='aggressive', dest='letter_aug_level',
                    help='AugLetterBank aug level: mild / normal / aggressive / extreme.')
    ap.add_argument('--letter-frozen', action='store_true', dest='letter_frozen',
                    help='FROZEN letter input: skip decoder, place letter bank '
                         'images directly via STN. Guarantees patches are real '
                         'letters; model only learns theta (placement). '
                         'Sweep24 LetterPlacer pattern.')
    ap.add_argument('--letter-samples-per-class', type=int, default=2000,
                    dest='letter_samples_per_class',
                    help='size of the PRE-AUGMENTED letter bank per class. '
                         '2000 × 26 = 52K letters cached once at startup; '
                         'runtime sample is a GPU gather (microseconds). '
                         'Per-step CPU augmentation made training 1000× slower.')
    ap.add_argument('--letter-loss-mode', default='l1', dest='letter_loss_mode',
                    choices=['l1', 'drift', 'both'],
                    help='Letter content loss flavor. l1 (default, vCurr) = L1 vs sampled real letter. '
                         'drift = D1 per-class drift (proposal eq, V_p=real same-class, V_q=gen same-class). '
                         'both = L1 (0.5x) + drift (0.5x).')
    ap.add_argument('--letter-drift-cp', type=int, default=64, dest='letter_drift_cp',
                    help='K_pos: real same-class letters per particle for D1 V_p.')
    ap.add_argument('--letter-drift-cn', type=int, default=8, dest='letter_drift_cn',
                    help='K_neg: other same-class GENERATED patches per particle for D1 V_q. '
                         'Self excluded. 0 = pure pull, no intra-class repulsion.')
    ap.add_argument('--letter-drift-R', default='0.005,0.02,0.1', dest='letter_drift_R',
                    help='comma-separated R_list for D1 drift. Scale-invariant (audit).')
    ap.add_argument('--letter-noise-aug', type=float, default=0.10, dest='letter_noise_aug',
                    help='noise_aug for D1 drift, applied symmetrically to gen+pos+neg.')
    ap.add_argument('--letter-drift-weight', type=float, default=0.1, dest='letter_drift_weight',
                    help='Sub-weight on drift branch inside the letter_weight curriculum. Drift '
                         'returns scaled MSE O(1) vs L1 O(0.1); default 0.1 compensates so '
                         'cur_letter_w stays comparable across modes. Bug-A fix per review.')
    ap.add_argument('--patch-tv', type=float, default=0.0, dest='patch_tv',
                    help='total-variation penalty on the 32x32 decoded glyph patches '
                         '(pre-STN). Directly smooths noisy letter content. Default 0 (off).')
    ap.add_argument('--patch-sharp', type=float, default=0.0, dest='patch_sharp',
                    help='sharpness penalty (push to {0,1}) on the decoded glyph patches. '
                         'Default 0 (off).')
    ap.add_argument('--letter-adapt', action='store_true', dest='letter_adapt',
                    help='dynamically weight the COMBINED letter loss (D1 drift + CE) so '
                         'that, in EMA, it stays at --letter-adapt-ratio times the face '
                         'pressure (D2 drift + region-IoU). Default off = fixed weights.')
    ap.add_argument('--letter-adapt-ratio', type=float, default=0.15, dest='letter_adapt_ratio',
                    help='target ratio D1drift:face for --letter-adapt, per-step (no EMA). '
                         'Favorite fixed-weight run sat at D1:face ~0.14.')
    ap.add_argument('--ce-adapt-ratio', type=float, default=0.05, dest='ce_adapt_ratio',
                    help='target ratio CE:face for --letter-adapt, adapted SEPARATELY from '
                         'D1. CE normally collapses to ~0; pinning it keeps the legibility '
                         'pressure constant. Keep modest (amplifying collapsed CE = noisier).')
    ap.add_argument('--letter-cls-weight', type=float, default=0.0, dest='letter_cls_weight',
                    help='Weight on the frozen-LetterCNN identity loss (deformation-invariant '
                         'legibility). ADDITIVE to D1 drift, not a replacement. 0 = off. '
                         'cross-entropy O(0-3); ~0.5-1.0 makes it comparable to other terms.')
    ap.add_argument('--letter-cls-ckpt', default='checkpoints/letter_cnn_32_mild.pt',
                    dest='letter_cls_ckpt',
                    help='Path to the pretrained LetterCNN state_dict (train via '
                         '.local/train_letter_cnn.py). Used only when letter-cls-weight > 0.')
    ap.add_argument('--drift-kernel', default='l2', dest='drift_kernel',
                    choices=['l2', 'l1', 'iou', 'dice', 'iou_ms', 'chamfer_l1',
                             'gradient', 'lowfreq', 'patch', 'iou_grad'],
                    help='Distance kernel for the D2 FACE composite drift. '
                         'sweep13 winning face recipe = iou. Default l2 (raw pixel).')
    ap.add_argument('--letter-drift-kernel', default='l2', dest='letter_drift_kernel',
                    choices=['l2', 'l1', 'iou', 'dice', 'iou_ms', 'chamfer_l1',
                             'gradient', 'lowfreq', 'patch', 'iou_grad'],
                    help='Distance kernel for the D1 per-LETTER drift, kept SEPARATE '
                         'from the face kernel. Default l2 preserves glyph shape; IoU '
                         'on letters mangles them into region-fill strokes (avoid).')
    ap.add_argument('--face-topk-pos', type=int, default=0, dest='face_topk_pos',
                    help='topk nearest positives in D2 face drift (sweep13 used 4).')
    ap.add_argument('--face-topk-neg', type=int, default=0, dest='face_topk_neg',
                    help='topk nearest negatives in D2 face drift (sweep13 used 8).')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=2000, dest='sample_every')
    ap.add_argument('--ckpt-every', type=int, default=2000, dest='ckpt_every',
                    help='save rolling G_ckpt.pt every N steps (atomic via .tmp rename). '
                         '0 disables. Default matches --sample-every.')
    ap.add_argument('--keep-ckpts', action='store_true', dest='keep_ckpts',
                    help='also persist a per-step G_step{N}.pt at every ckpt_every '
                         '(keep ALL, not rolling) so the early-peak ckpt is recoverable. '
                         'The model peaks early then over-trains -> needed to lock best.')

    # Transformer-specific knobs
    ap.add_argument('--d-token', type=int, default=256, dest='d_token')
    ap.add_argument('--n-layers', type=int, default=4, dest='n_layers',
                    help='total transformer layers; split equally stage1/stage2. Must be even.')
    ap.add_argument('--n-heads', type=int, default=8, dest='n_heads')
    ap.add_argument('--patch-size', type=int, default=32, dest='patch_size')
    ap.add_argument('--r-min', type=float, default=0.10, dest='r_min')
    ap.add_argument('--r-max', type=float, default=0.55, dest='r_max')
    ap.add_argument('--dxy-max', type=float, default=0.6, dest='dxy_max')
    ap.add_argument('--no-rotation', action='store_true',
                    help='disable affine rotation (rot forced to 0). Useful as a warm-up '
                         'phase before flipping rotation back on.')
    ap.add_argument('--alpha-composite', action='store_true', dest='alpha_composite',
                    help='CHANGE A: softmax-over-slots alpha compositing. Decoder emits a '
                         '2nd alpha channel; slots compete per-pixel (Slot-Attention style) '
                         'with a learnable background alpha so empty regions stay empty. '
                         'Default OFF -> sum().clamp() composition (bit-identical to prior runs). '
                         'DEPRECATED: per-pixel alpha is a cheating free mask; use --slot-intensity.')
    ap.add_argument('--slot-intensity', action='store_true', dest='slot_intensity',
                    help='LEGIT composite: per-slot UNIFORM gain in [0,1] (one scalar per '
                         'layer, same value every pixel), canvas = clamp(sum_k gain_k*layer_k). '
                         'A slot can globally dim/brighten its whole letter but CANNOT carve '
                         'per-pixel -> the face must be formed by real letter strokes + STN '
                         'placement, not a free per-pixel mask. Takes precedence over --alpha-composite.')

    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
