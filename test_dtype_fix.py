#!/usr/bin/env python3
"""Test script to verify the dtype fix for wanda_selectivity.py"""

import torch
import sys
import os

# Add the lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

try:
    from wanda_selectivity import SelectivityStatsLight
    print("✅ Successfully imported SelectivityStatsLight")
    
    # Test with different dtypes that could cause the issue
    test_dtypes = [torch.float16, torch.int32, torch.long, torch.float32]
    
    for dtype in test_dtypes:
        print(f"\n🧪 Testing with dtype: {dtype}")
        
        # Create test data
        stats = SelectivityStatsLight(num_channels=512)
        
        # Create sample data with the problematic dtype
        test_data = torch.randn(100, 512, dtype=torch.float32)
        if dtype != torch.float32:
            if dtype == torch.int32:
                test_data = (test_data * 100).int()
            elif dtype == torch.long:
                test_data = (test_data * 100).long()
            elif dtype == torch.float16:
                test_data = test_data.half()
        
        print(f"   Input dtype: {test_data.dtype}")
        
        # Update stats
        stats.update(test_data)
        
        # Try to finalize (this would fail before the fix)
        try:
            idf_scores, spikiness_scores = stats.finalize()
            print(f"   ✅ Success! IDF shape: {idf_scores.shape}, Spikiness shape: {spikiness_scores.shape}")
            print(f"   IDF dtype: {idf_scores.dtype}, Spikiness dtype: {spikiness_scores.dtype}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n🎉 All tests completed!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")