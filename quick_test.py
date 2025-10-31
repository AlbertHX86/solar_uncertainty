"""
快速测试脚本 - 验证环境和代码
使用少量epochs快速测试所有模型
"""
import os
import sys

# OpenMP设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 临时修改配置为快速测试模式
print("="*80)
print("快速测试模式 - 使用少量epochs验证代码")
print("="*80)

from config import config

# 覆盖配置（快速测试）
config.epochs_baseline = 2
config.epochs_quantile = 2
config.epochs_gp = 2
config.epochs_caun_stage1 = 2
config.epochs_caun_stage2 = 2
config.batch_size = 32

print(f"\n[测试配置]")
print(f"  Epochs: {config.epochs_baseline} (大大减少)")
print(f"  Batch size: {config.batch_size}")
print(f"  预计运行时间: 3-5分钟\n")

# 运行主程序
import main

if __name__ == "__main__":
    print("\n开始快速测试...")
    try:
        results, metrics = main.main()
        
        print("\n" + "="*80)
        print("✓ 快速测试成功！")
        print("="*80)
        print("\n所有模型和流程验证通过。")
        print("如需完整训练，请运行: python main.py")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ 测试失败")
        print("="*80)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

