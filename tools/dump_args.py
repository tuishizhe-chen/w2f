"""Extract args dict from a saved G_final.pt and print as a command line."""
import sys, torch, json
ck = torch.load(sys.argv[1], weights_only=False, map_location='cpu')
args = ck.get('args', {})
print('--- raw args dict ---')
print(json.dumps(args, indent=2, default=str))
print('--- as cli flags ---')
parts = []
for k, v in args.items():
    if v is None or v == 0 or v == 0.0 or v is False:
        continue
    flag = '--' + k.replace('_', '-')
    if isinstance(v, bool):
        parts.append(flag)
    else:
        parts.append(f'{flag} {v}')
print(' '.join(parts))
