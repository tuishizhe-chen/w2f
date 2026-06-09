import sys
sys.path.insert(0, '/root/w2f/src')
import face_drift_pixel
import drift_loss
print('imports OK')
print('drift_loss signature first 14 vars:', drift_loss.drift_loss.__code__.co_varnames[:14])
# check that the PixelGen has forward_with_logits
import inspect
assert hasattr(face_drift_pixel.PixelGen, 'forward_with_logits'), 'missing forward_with_logits'
sig = inspect.signature(face_drift_pixel.PixelGen.__init__)
assert 'head_refine' in sig.parameters, 'missing head_refine ctor arg'
print('PixelGen ctor params:', list(sig.parameters))
print('all checks pass')
