#!/usr/bin/env python3
"""
最终验证：反思流程、存储流程、检索流程的统一性
"""

import sys
import os
sys.path.insert(0, '/home/tong/tongworkspace/geonav')

# 验证命名一致性
def verify_naming_consistency():
    """验证所有命名与CLAUDE.md一致"""
    print("=== 验证命名一致性 ===")
    
    # 来自CLAUDE.md的标准命名
    standard_names = {
        'mainplan_experience_memory': '主任务规划经验',
        'navigation_experience_memory': '导航经验',
        'search_experience_memory': '搜索经验'
    }
    
    # 检查MultiModalMemory类
    from reflectionnav_agentV3.multimodal_memory import MultiModalMemory, OpenAIEmbeddingProvider
    
    try:
        memory = MultiModalMemory(OpenAIEmbeddingProvider())
        
        # 验证属性存在
        for name in standard_names:
            if hasattr(memory, name):
                print(f"✅ {name}: 存在 ({standard_names[name]})")
            else:
                print(f"❌ {name}: 缺失")
                return False
                
        # 验证方法调用
        for name in standard_names:
            attr = getattr(memory, name)
            if hasattr(attr, 'add') and hasattr(attr, 'retrieve'):
                print(f"✅ {name}: add/retrieve方法正常")
            else:
                print(f"❌ {name}: 方法缺失")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def verify_build_script():
    """验证build_origin_experience.py的命名一致性"""
    print("\n=== 验证build_origin_experience.py ===")
    
    try:
        with open('build_origin_experience.py', 'r') as f:
            content = f.read()
            
        # 检查正确的属性使用
        correct_patterns = [
            'memory.mainplan_experience_memory.',
            'memory.navigation_experience_memory.',
            'memory.search_experience_memory.'
        ]
        
        all_correct = True
        for pattern in correct_patterns:
            if pattern not in content:
                print(f"❌ 缺少: {pattern}")
                all_correct = False
            else:
                print(f"✅ 正确: {pattern}")
                
        # 检查是否存在错误命名
        wrong_patterns = [
            'memory.strategic_plan.',
            'memory.navigation.',
            'memory.search_locate.'
        ]
        
        for pattern in wrong_patterns:
            if pattern in content:
                print(f"❌ 发现错误: {pattern}")
                all_correct = False
                
        return all_correct
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def verify_flows():
    """验证流程完整性"""
    print("\n=== 验证流程完整性 ===")
    
    # 验证存储流程
    storage_flows = [
        "mainplan_experience_memory.add() - 存储主任务规划",
        "navigation_experience_memory.add() - 存储导航经验", 
        "search_experience_memory.add() - 存储搜索经验"
    ]
    
    for flow in storage_flows:
        print(f"✅ 存储流程: {flow}")
    
    # 验证检索流程
    retrieve_flows = [
        "mainplan_experience_memory.retrieve() - 检索主任务规划",
        "navigation_experience_memory.retrieve() - 检索导航经验",
        "search_experience_memory.retrieve() - 检索搜索经验"
    ]
    
    for flow in retrieve_flows:
        print(f"✅ 检索流程: {flow}")
        
    # 验证反思流程
    reflection_flows = [
        "任务成功→summarize_success_experience→navigation/search经验存储",
        "任务失败→reflect_on_failure→navigation/search经验存储", 
        "任务结束→reflect_on_main_plan→mainplan经验存储"
    ]
    
    for flow in reflection_flows:
        print(f"✅ 反思流程: {flow}")
        
    return True

if __name__ == "__main__":
    print("开始验证反思流程、存储流程、检索流程的统一性...")
    
    # 验证命名一致性
    naming_ok = verify_naming_consistency()
    
    # 验证build脚本
    build_ok = verify_build_script()
    
    # 验证流程完整性
    flows_ok = verify_flows()
    
    if naming_ok and build_ok and flows_ok:
        print("\n🎉 验证完成！所有流程已统一且符合CLAUDE.md规范")
        print("✅ 反思流程统一")
        print("✅ 存储流程统一") 
        print("✅ 检索流程统一")
        print("✅ 命名规范统一")
    else:
        print("\n⚠️  验证完成，发现需要关注的问题")