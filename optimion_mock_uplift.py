# coding=utf8
import argparse
import logging
import json
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model

logging.basicConfig(filename='casual_inference_miaosuan.log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")
logger = logging.getLogger('urbanGUI')


# mock data
def generate_data():
    a = np.random.rand(50, 3).astype('float16')
    data = pd.DataFrame(a, columns=["outcome0", "outcome1", "outcome2"])
    # 模拟红包的金额
    red_packet = [0, 2, 5]
    data["uplift0"] = 0
    data["uplift1"] = (data["outcome1"] - data["outcome0"]) / (red_packet[1] - red_packet[0])  # 使用斜率代替uplift
    data["uplift2"] = (data["outcome2"] - data["outcome1"]) / (red_packet[2] - red_packet[1])  # 使用斜率代替uplift

    # 理论上预测每个红包的概率
    outcome = data[["outcome0", "outcome1", "outcome2"]].values
    outcome = outcome.tolist()

    ## uplift
    costs = data[["uplift0", "uplift1", "uplift2"]].values
    costs = costs.tolist()

    return outcome, costs


def optimization_define(outcome, costs, red_limit=10, red_packet=[0, 2, 5]):
    num_workers = len(costs)
    num_tasks = len(costs[0])
    model = cp_model.CpModel()

    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = model.NewBoolVar(f'x[{worker},{task}]')

    # Each worker is assigned to at most one task.
    # 每个用户只能发一个红包
    for worker in range(num_workers):
        model.AddAtMostOne(x[worker, task] for task in range(num_tasks))

    all_red = []
    for worker in range(num_workers):
        for task in range(num_tasks):
            all_red.append(int(outcome[worker][task] * 100) * red_packet[task] * x[worker, task])  # *100是因为该求解器值支持整数
    model.Add(sum(all_red) <= (red_limit * 100))  # 限制条件也乘以100

    objective_terms = []
    for worker in range(num_workers):
        for task in range(num_tasks):
            objective_terms.append(costs[worker][task] * x[worker, task])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Total cost = {solver.ObjectiveValue()}\n')
        for worker in range(num_workers):
            for task in range(num_tasks):
                if solver.BooleanValue(x[worker, task]):
                    print(f'Worker {worker} assigned to task {task}.' +
                          f' Cost = {costs[worker][task]}')
    else:
        print('No solution found.')


if __name__ == "__main__":
    red_limit = 100
    red_packet = [0, 6, 10]
    outcome, costs = generate_data()
    optimization_define(outcome, costs, red_limit, red_packet)
