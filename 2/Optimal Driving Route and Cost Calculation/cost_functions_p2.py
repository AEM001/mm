import numpy as np
import heapq
import matplotlib.pyplot as plt
from road_network_p2 import *

# === 三）罚款函数与检测概率 ===

def calculate_fine(speed_limit, over_pct):
    """
    计算超速罚款金额
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        罚款金额（元）
    """
    if over_pct <= 0:
        return 0
    
    # 将小数形式的百分比转换为整数百分比
    over_pct_int = over_pct * 100
    
    # 根据限速和超速百分比确定罚款金额
    if speed_limit < 50:
        # 低于50 km/h
        if over_pct_int <= 20:
            return 50
        elif over_pct_int <= 50:
            return 100
        elif over_pct_int <= 70:
            return 300
        else:
            return 500
    elif speed_limit <= 80:
        # 50-80 km/h
        if over_pct_int <= 20:
            return 100
        elif over_pct_int <= 50:
            return 150
        elif over_pct_int <= 70:
            return 500
        else:
            return 1000
    elif speed_limit <= 100:
        # 80-100 km/h
        if over_pct_int <= 20:
            return 150
        elif over_pct_int <= 50:
            return 200
        elif over_pct_int <= 70:
            return 1000
        else:
            return 1500
    else:
        # 高于100 km/h
        if over_pct_int <= 50:
            return 200
        elif over_pct_int <= 70:
            return 1500
        else:
            return 2000

def detection_probability(over_pct):
    """
    计算超速被任意一个雷达探测到的概率
    
    参数:
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        被探测到的概率（0.0-1.0）
    """
    if over_pct <= 0:
        return 0.0
    elif over_pct <= 0.2:
        return 0.7
    elif over_pct <= 0.5:
        return 0.9
    elif over_pct <= 0.7:
        return 0.99
    else:
        return 1.0  # 超过70%按100%概率处理

def expected_fine(speed_limit, over_pct, has_fixed_radar):
    """
    计算期望罚款
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        has_fixed_radar: 是否有固定雷达
        
    返回:
        期望罚款金额（元）
    """
    if over_pct <= 0:
        return 0.0
    
    fine_amount = calculate_fine(speed_limit, over_pct)
    prob = detection_probability(over_pct)
    
    # 考虑移动雷达的存在（20个移动雷达可能出现在任何路段）
    # 简化处理：假设每个路段有相同概率遇到移动雷达
    # 实际上应该是一个更复杂的概率模型，这里做了简化
    return fine_amount * prob

def calculate_segment_cost(speed_limit, over_pct, has_fixed_radar):
    """
    计算路段总成本（包括时间成本、燃油费、通行费和期望罚款）
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        has_fixed_radar: 是否有固定雷达
        
    返回:
        (总成本, 行驶时间)
    """
    # 实际速度
    actual_speed = speed_limit * (1 + over_pct)
    
    # 行驶时间（小时）
    travel_time = L / actual_speed
    
    # 时间成本
    time_cost = time_cost_rate * travel_time
    
    # 燃油费
    fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)  # 油耗（升）
    fuel_cost = fuel_consumption * fuel_price
    
    # 高速通行费
    toll_cost = toll_per_km * L if speed_limit == 120 else 0.0
    
    # 期望罚款
    fine_cost = expected_fine(speed_limit, over_pct, has_fixed_radar)
    
    # 总成本
    total_cost = time_cost + fuel_cost + toll_cost + fine_cost
    
    return total_cost, travel_time

# 测试费用函数
def test_cost_functions():
    """测试费用计算函数"""
    print("\n=== 测试费用计算函数 ===")
    
    # 测试罚款计算
    print("\n1. 罚款金额测试:")
    test_cases = [
        (40, 0.1, "50元"),
        (40, 0.3, "100元"),
        (40, 0.6, "300元"),
        (60, 0.1, "100元"),
        (60, 0.3, "150元"),
        (60, 0.6, "500元"),
        (90, 0.1, "150元"),
        (90, 0.3, "200元"),
        (90, 0.6, "1000元"),
        (120, 0.1, "200元"),
        (120, 0.3, "200元"),
        (120, 0.6, "1500元")
    ]
    
    for speed_limit, over_pct, expected in test_cases:
        fine = calculate_fine(speed_limit, over_pct)
        print(f"限速{speed_limit}km/h, 超速{over_pct*100}%, 罚款: {fine}元 (预期: {expected})")
    
    # 测试检测概率
    print("\n2. 检测概率测试:")
    test_cases = [
        (0.0, "0%"),
        (0.1, "70%"),
        (0.2, "70%"),
        (0.3, "90%"),
        (0.5, "90%"),
        (0.6, "99%"),
        (0.7, "99%")
    ]
    
    for over_pct, expected in test_cases:
        prob = detection_probability(over_pct)
        print(f"超速{over_pct*100}%, 检测概率: {prob*100}% (预期: {expected})")
    
    # 测试期望罚款
    print("\n3. 期望罚款测试:")
    test_cases = [
        (60, 0.0, False, "0元"),
        (60, 0.2, False, "100元 * 70% = 70元"),
        (90, 0.5, True, "200元 * 90% = 180元"),
        (120, 0.7, True, "1500元 * 99% = 1485元")
    ]
    
    for speed_limit, over_pct, has_radar, expected in test_cases:
        exp_fine = expected_fine(speed_limit, over_pct, has_radar)
        print(f"限速{speed_limit}km/h, 超速{over_pct*100}%, 固定雷达: {has_radar}, 期望罚款: {exp_fine}元 (预期: {expected})")
    
    # 测试路段总成本
    print("\n4. 路段总成本测试:")
    test_cases = [
        (60, 0.0),
        (60, 0.2),
        (90, 0.5),
        (120, 0.7)
    ]
    
    for speed_limit, over_pct in test_cases:
        has_radar = speed_limit >= 90
        total_cost, travel_time = calculate_segment_cost(speed_limit, over_pct, has_radar)
        actual_speed = speed_limit * (1 + over_pct)
        
        # 计算各项成本
        t = L / actual_speed
        time_cost = time_cost_rate * t
        fuel = (0.0625 * actual_speed + 1.875) * (L / 100.0)
        fuel_cost = fuel * fuel_price
        toll_cost = toll_per_km * L if speed_limit == 120 else 0.0
        fine_cost = expected_fine(speed_limit, over_pct, has_radar)
        
        print(f"\n限速{speed_limit}km/h, 超速{over_pct*100}%, 实际速度{actual_speed}km/h:")
        print(f"  行驶时间: {travel_time:.4f}小时")
        print(f"  时间成本: {time_cost:.2f}元")
        print(f"  燃油费: {fuel_cost:.2f}元")
        print(f"  通行费: {toll_cost:.2f}元")
        print(f"  期望罚款: {fine_cost:.2f}元")
        print(f"  总成本: {total_cost:.2f}元")

# 运行测试
test_cost_functions()

# 超速选项
speed_options = [0.0, 0.2, 0.5, 0.7]  # 0%, 20%, 50%, 70%

print("\n费用函数设计完成，准备实现多标记Dijkstra算法...")
