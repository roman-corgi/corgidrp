# test_pol_layer_fix.py
from corgidrp.mocks import create_default_L3_headers
from corgidrp.data import Image
import numpy as np

print("="*60)
print("TESTING POL FIX - L3 Headers")
print("="*60)

# Get headers
prihdr, exthdr, errhdr, dqhdr = create_default_L3_headers()
print(f"ERROR HEADER has LAYER_1: {errhdr.get('LAYER_1', 'NOT FOUND')}")
print("="*60 + "\n")

# Test 1: WITHOUT fix (old buggy way)
print("TEST 1: WITHOUT err_hdr (BEFORE FIX)")
print("="*60)
stellar_sys_wp_data = np.zeros((2, 54, 54))
img_before = Image(stellar_sys_wp_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
if img_before.err_hdr is not None:
    print(f"Image err_hdr has LAYER_1: {img_before.err_hdr.get('LAYER_1', 'NOT FOUND')}")
else:
    print("Image err_hdr is None!")
print("="*60 + "\n")

# Test 2: WITH fix (new correct way)
print("TEST 2: WITH err_hdr (AFTER FIX)")
print("="*60)
stellar_nd_wp_data = np.zeros((2, 54, 54))
img_after = Image(stellar_nd_wp_data, 
                  pri_hdr=prihdr.copy(), 
                  ext_hdr=exthdr.copy(),
                  err_hdr=errhdr.copy(),
                  dq_hdr=dqhdr.copy())
print(f"Image err_hdr has LAYER_1: {img_after.err_hdr.get('LAYER_1', 'NOT FOUND')}")
print("="*60 + "\n")

# Summary
print("SUMMARY:")
print("="*60)
if img_before.err_hdr is None or img_before.err_hdr.get('LAYER_1') != 'combined_error':
    print("✗ BEFORE FIX: LAYER_1 missing (as expected)")
else:
    print("✓ BEFORE FIX: LAYER_1 present (unexpected!)")

if img_after.err_hdr.get('LAYER_1') == 'combined_error':
    print("✓ AFTER FIX: LAYER_1 = 'combined_error' (CORRECT!)")
else:
    print("✗ AFTER FIX: LAYER_1 missing (something wrong)")
print("="*60)