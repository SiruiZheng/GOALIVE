import numpy as np
import torch
import random
import torch.nn.functional as F
import pandas as pd

# setup the env
L = 5
H = 9
goal_star = np.array([1, 1])
transit_begin = np.array([1, 1])
transit_end = np.array([1, 3])
start = np.array([3, 3])
action_set = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])


class Env:
    def __init__(self, H, L, transit_begin, transit_end, start):
        self.H = H
        self.L = L
        self.transit_begin = transit_begin
        self.transit_end = transit_end
        self.start = start

    def step(self, state, action, goal):
        # given the state-action pair, returnn the coresponding state and action
        # 01 up 0-1 down 10 right -10 left
        if (state - self.transit_begin).any():
            state_next = state + action
        else:
            state_next = self.transit_end
        state_next_cl = np.clip(state_next, 0, 4)
        state_next_cl = state_next_cl.reshape((2))
        if not (state - goal).any():
            return (state_next_cl, 5)
        elif (state_next_cl - state_next).any():
            return (state_next_cl, -1)
        return (state_next_cl, 0)


env = Env(H, L, transit_begin, transit_end, start)


# Q net
class QNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(QNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden[0])
        self.hidden2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        self.predict = torch.nn.Linear(n_hidden[1], n_output, bias=False)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except AttributeError:
            1 + 1


all_goal = []
for i in range(L):
    for j in range(L):
        all_goal.append([i, j])
# training
dataset = [[] for i in range(H)]
N = 100
T = 100
eps_grd = 0.05
repeat_sp = 1
test_each = 1
overall_iter = 5
overall_out = [[] for oit in range(overall_iter)]
for oit in range(overall_iter):
    dataset = [[] for i in range(H)]
    qnet = [QNet(n_feature=4, n_hidden=(100, 100), n_output=4) for h in range(H)]
    qnet = [item.apply(init_weights) for item in qnet]
    for n in range(N):
        # collect traj
        for rsp in range(repeat_sp):
            state = start
            for h in range(H):
                input_Q = torch.FloatTensor(np.concatenate([state, goal_star]))
                out = qnet[h](input_Q)
                if np.random.uniform() < eps_grd:
                    action_id = random.choice([0, 1, 2, 3])
                else:
                    action_id = torch.argmax(out).item()
                action = action_set[action_id]
                state_next, reward = env.step(state, action, goal_star)
                # print(state_next)
                # print(goal_star)
                input_next_Q = torch.FloatTensor(
                    np.concatenate([state_next, goal_star])
                )
                if h + 1 < H:
                    out_next = qnet[h + 1](input_next_Q)
                dataset[h].append((state, action_id, state_next))
                """print(
                    "n:{}\t h:{}\t state:{}\t action:{}\t next state:{}\t estimated Q:{}\t estimated Q next:{}".format(
                        n,
                        h,
                        state,
                        action,
                        state_next,
                        np.round(out.detach().numpy(), decimals=3),
                        np.round(out_next.detach().numpy(), decimals=3),
                    )
                )"""
                state = state_next
        # use traj for training
        for h_bar in range(H):
            h = H - h_bar - 1
            optimizer = torch.optim.Adam(qnet[h].parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()
            input_Q = np.concatenate(
                [
                    np.concatenate([v[0], u]).reshape((1, -1))
                    for u in all_goal
                    for v in dataset[h]
                ],
                axis=0,
            )
            output_Q = np.concatenate(
                [
                    np.concatenate([v[2], u]).reshape((1, -1))
                    for u in all_goal
                    for v in dataset[h]
                ],
                axis=0,
            )
            action_Q = torch.LongTensor(
                [v[1] for u in all_goal for v in dataset[h]]
            ).reshape((-1))
            reward_hit = [
                float((v[0] - v[2]).any()) - 1 for u in all_goal for v in dataset[h]
            ]
            reward_bn = [
                5 - float((v[0] - u).any()) * 5 for u in all_goal for v in dataset[h]
            ]
            reward_batch = torch.FloatTensor(reward_hit) + torch.FloatTensor(reward_bn)
            n_dt = len(action_Q)
            if h == H - 1:
                next_Q = torch.zeros([n_dt])
            else:
                next_Q = torch.max(
                    qnet[h + 1](torch.FloatTensor(output_Q)), axis=1
                ).values
            for t in range(T):
                # print(next_Q.shape)
                # print(input_Q)
                Q_current = qnet[h](torch.FloatTensor(input_Q))
                # print(qnet[h](torch.FloatTensor(input_Q)))
                Q_current = torch.gather(Q_current, 1, action_Q.view(-1, 1)).squeeze()
                diff = reward_batch + next_Q - Q_current
                # print(diff)
                # print(Q_current)
                # print(diff)
                # print(output_Q)
                # print(diff)
                diff = diff.reshape((n + 1, L * L, repeat_sp))
                # print(diff)
                diff = torch.sum(diff, axis=2)
                # print(diff)
                big_bell = torch.max(torch.abs(diff), axis=1).values
                # print(big_bell)
                loss = torch.mean(torch.pow(big_bell, 2))  # 计算两者的误差
                # print(loss)
                # print(reward_batch + next_Q)
                # print(Q_current)
                optimizer.zero_grad()  # 清n空上一步的残余更新参数值
                loss.backward(retain_graph=True)  # 误差反向传播, 计算参数更新值
                if loss < 0.001:
                    break
                normm = torch.nn.utils.clip_grad_norm_(qnet[h].parameters(), 20)
                """
                if t % (T//10) == 0:
                    print('a')
                    print(input_Q[-repeat_sp:],action_set[action_Q[-repeat_sp:]],output_Q[-repeat_sp:])
                    print('Q_now:{}\t reward:{}\t Q_next:{}'.format(Q_current[-repeat_sp:],reward_batch[-repeat_sp:],next_Q[-repeat_sp:]))
                    print("n:{}\t h:{}\t t:{}\t loss:{:.3f}".format(n, h, t, loss))
                    print(normm)
                """
                optimizer.step()
        # test policy
        reward_test = 0
        for tc in range(test_each):
            state = start
            for h in range(H):
                input_Q = torch.FloatTensor(np.concatenate([state, goal_star]))
                out = qnet[h](input_Q)
                action_id = torch.argmax(out).item()
                action = action_set[action_id]
                state_next, reward = env.step(state, action, goal_star)
                reward_test += reward
                # print(state_next)
                # print(goal_star)
                input_next_Q = torch.FloatTensor(
                    np.concatenate([state_next, goal_star])
                )
                if h + 1 < H:
                    out_next = qnet[h + 1](input_next_Q)
                # dataset[h].append((state, action_id, state_next))
                state = state_next
        print("testing n:{}\t reward:{:.2f}\t ".format(n, reward_test / test_each))
        overall_out[oit].append(reward_test / test_each)
pd.DataFrame(np.array(overall_out).T).to_csv(
    "out_GOALIVE.csv", header=False, index=False
)
# generate Q
# generate next Q
# optimize
# single goal


overall_out_single = [[] for oit in range(overall_iter)]
for oit in range(overall_iter):
    qnet_single = [QNet(n_feature=2, n_hidden=(100, 100), n_output=4) for h in range(H)]
    qnet_single = [item.apply(init_weights) for item in qnet_single]

    # training
    dataset = [[] for i in range(H)]
    for n in range(N):
        # collect traj
        for rsp in range(repeat_sp):
            state = start
            for h in range(H):
                input_Q = torch.FloatTensor([state])
                out = qnet_single[h](input_Q)
                if np.random.uniform() < eps_grd:
                    action_id = random.choice([0, 1, 2, 3])
                else:
                    action_id = torch.argmax(out).item()
                action = action_set[action_id]
                state_next, reward = env.step(state, action, goal_star)
                # print(state_next)
                # print(goal_star)
                input_next_Q = torch.FloatTensor([state_next])
                if h + 1 < H:
                    out_next = qnet_single[h + 1](input_next_Q)
                dataset[h].append((state, action_id, state_next))
                """print(
                    "n:{}\t h:{}\t state:{}\t action:{}\t next state:{}\t estimated Q:{}\t estimated Q next:{}".format(
                        n,
                        h,
                        state,
                        action,
                        state_next,
                        np.round(out.detach().numpy(), decimals=3),
                        np.round(out_next.detach().numpy(), decimals=3),
                    )
                )"""
                state = state_next
        # use traj for training
        for h_bar in range(H):
            h = H - h_bar - 1
            optimizer = torch.optim.Adam(qnet_single[h].parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()
            input_Q = np.concatenate(
                [[v[0]] for v in dataset[h]],
                axis=0,
            )
            # pri
            output_Q = np.concatenate(
                [[v[2]] for v in dataset[h]],
                axis=0,
            )
            action_Q = torch.LongTensor([v[1] for v in dataset[h]]).reshape((-1))
            reward_hit = [float((v[0] - v[2]).any()) - 1 for v in dataset[h]]
            reward_bn = [5 - float((v[0] - goal_star).any()) * 5 for v in dataset[h]]
            reward_batch = torch.FloatTensor(reward_hit) + torch.FloatTensor(reward_bn)
            n_dt = len(action_Q)
            if h == H - 1:
                next_Q = torch.zeros([n_dt])
            else:
                next_Q = torch.max(
                    qnet_single[h + 1](torch.FloatTensor(output_Q)), axis=1
                ).values
            for t in range(T):
                # print(next_Q.shape)
                # print(input_Q)
                Q_current = qnet_single[h](torch.FloatTensor(input_Q))
                # print(qnet_single[h](torch.FloatTensor(input_Q)))
                Q_current = torch.gather(Q_current, 1, action_Q.view(-1, 1)).squeeze()
                diff = reward_batch + next_Q - Q_current
                # print(diff)
                # print(Q_current)
                # print(diff)
                # print(output_Q)
                # print(diff)
                # diff = diff.reshape((n + 1, L * L, repeat_sp))
                # print(diff)
                # diff = torch.sum(diff, axis=2)
                # print(diff)
                # big_bell = torch.max(torch.abs(diff), axis=1).values
                # print(big_bell)
                loss = torch.mean(torch.pow(diff, 2))  # 计算两者的误差
                # print(loss)
                # print(reward_batch + next_Q)
                # print(Q_current)
                optimizer.zero_grad()  # 清n空上一步的残余更新参数值
                loss.backward(retain_graph=True)  # 误差反向传播, 计算参数更新值
                if loss < 0.001:
                    break
                normm = torch.nn.utils.clip_grad_norm_(qnet_single[h].parameters(), 20)
                """
                if t % (T//10) == 0:
                    print('a')
                    print(input_Q[-repeat_sp:],action_set[action_Q[-repeat_sp:]],output_Q[-repeat_sp:])
                    print('Q_now:{}\t reward:{}\t Q_next:{}'.format(Q_current[-repeat_sp:],reward_batch[-repeat_sp:],next_Q[-repeat_sp:]))
                    print("n:{}\t h:{}\t t:{}\t loss:{:.3f}".format(n, h, t, loss))
                    print(normm)
                """
                optimizer.step()
        # test policy
        reward_test = 0
        for tc in range(test_each):
            state = start
            for h in range(H):
                input_Q = torch.FloatTensor([state])
                out = qnet_single[h](input_Q)
                action_id = torch.argmax(out).item()
                action = action_set[action_id]
                state_next, reward = env.step(state, action, goal_star)
                reward_test += reward
                # print(state_next)
                # print(goal_star)
                input_next_Q = torch.FloatTensor([state_next])
                if h + 1 < H:
                    out_next = qnet_single[h + 1](input_next_Q)
                # dataset[h].append((state, action_id, state_next))
                state = state_next
        print("testing n:{}\t reward:{:.2f}\t ".format(n, reward_test / test_each))
        overall_out_single[oit].append(reward_test / test_each)

        # generate Q
        # generate next Q
        # optimize
pd.DataFrame(np.array(overall_out_single).T).to_csv(
    "out_single.csv", header=False, index=False
)

# HER


overall_out_HER = [[] for oit in range(overall_iter)]
for oit in range(overall_iter):
    qnet_HER = [QNet(n_feature=4, n_hidden=(100, 100), n_output=4) for h in range(H)]
    qnet_HER = [item.apply(init_weights) for item in qnet_HER]

    # training
    dataset = [[] for i in range(H)]
    for n in range(N):
        # collect traj
        for rsp in range(repeat_sp):
            state = start
            for h in range(H):
                input_Q = torch.FloatTensor(np.concatenate([state, goal_star]))
                out = qnet_HER[h](input_Q)
                if np.random.uniform() < eps_grd:
                    action_id = random.choice([0, 1, 2, 3])
                else:
                    action_id = torch.argmax(out).item()
                action = action_set[action_id]
                state_next, reward = env.step(state, action, goal_star)
                # print(state_next)
                # print(goal_star)
                input_next_Q = torch.FloatTensor(
                    np.concatenate([state_next, goal_star])
                )
                if h + 1 < H:
                    out_next = qnet_HER[h + 1](input_next_Q)
                dataset[h].append((state, action_id, state_next))
                """print(
                    "n:{}\t h:{}\t state:{}\t action:{}\t next state:{}\t estimated Q:{}\t estimated Q next:{}".format(
                        n,
                        h,
                        state,
                        action,
                        state_next,
                        np.round(out.detach().numpy(), decimals=3),
                        np.round(out_next.detach().numpy(), decimals=3),
                    )
                )"""
                state = state_next
        # use traj for training
        for h_bar in range(H):
            h = H - h_bar - 1
            optimizer = torch.optim.Adam(qnet_HER[h].parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()
            input_Q = np.concatenate(
                [
                    np.concatenate([dataset[h][v][0], dataset[h + u][v][2]]).reshape(
                        (1, -1)
                    )
                    for u in range(h_bar + 1)
                    for v in range(len(dataset[h]))
                ],
                axis=0,
            )
            # pri
            output_Q = np.concatenate(
                [
                    np.concatenate([dataset[h][v][2], dataset[h + u][v][2]]).reshape(
                        (1, -1)
                    )
                    for u in range(h_bar + 1)
                    for v in range(len(dataset[h]))
                ],
                axis=0,
            )
            action_Q = torch.LongTensor(
                [v[1] for u in range(h_bar + 1) for v in dataset[h]]
            ).reshape((-1))
            reward_hit = [
                float((v[0] - v[2]).any()) - 1
                for u in range(h_bar + 1)
                for v in dataset[h]
            ]
            reward_bn = [
                5 - float((dataset[h][v][0] - dataset[h + u][v][2]).any()) * 5
                for u in range(h_bar + 1)
                for v in range(len(dataset[h]))
            ]
            reward_batch = torch.FloatTensor(reward_hit) + torch.FloatTensor(reward_bn)
            n_dt = len(action_Q)
            if h == H - 1:
                next_Q = torch.zeros([n_dt])
            else:
                next_Q = torch.max(
                    qnet_HER[h + 1](torch.FloatTensor(output_Q)), axis=1
                ).values
            for t in range(T):
                # print(next_Q.shape)
                # print(input_Q)
                Q_current = qnet_HER[h](torch.FloatTensor(input_Q))
                # print(qnet_HER[h](torch.FloatTensor(input_Q)))
                Q_current = torch.gather(Q_current, 1, action_Q.view(-1, 1)).squeeze()
                diff = reward_batch + next_Q - Q_current
                # print(diff)
                # print(Q_current)
                # print(diff)
                # print(output_Q)
                # print(diff)
                # diff = diff.reshape((n + 1, L * L, repeat_sp))
                # print(diff)
                # diff = torch.sum(diff, axis=2)
                # print(diff)
                # big_bell = torch.max(torch.abs(diff), axis=1).values
                # print(big_bell)
                loss = torch.mean(torch.pow(diff, 2))  # 计算两者的误差
                # print(loss)
                # print(reward_batch + next_Q)
                # print(Q_current)
                optimizer.zero_grad()  # 清n空上一步的残余更新参数值
                loss.backward(retain_graph=True)  # 误差反向传播, 计算参数更新值
                if loss < 0.001:
                    break
                normm = torch.nn.utils.clip_grad_norm_(qnet_HER[h].parameters(), 20)
                """
                if t % (T//10) == 0:
                    print('a')
                    print(input_Q[-repeat_sp:],action_set[action_Q[-repeat_sp:]],output_Q[-repeat_sp:])
                    print('Q_now:{}\t reward:{}\t Q_next:{}'.format(Q_current[-repeat_sp:],reward_batch[-repeat_sp:],next_Q[-repeat_sp:]))
                    print("n:{}\t h:{}\t t:{}\t loss:{:.3f}".format(n, h, t, loss))
                    print(normm)
                """
                optimizer.step()
        # test policy
        reward_test = 0
        for tc in range(test_each):
            state = start
            for h in range(H):
                input_Q = torch.FloatTensor(np.concatenate([state, goal_star]))
                out = qnet_HER[h](input_Q)
                action_id = torch.argmax(out).item()
                action = action_set[action_id]
                state_next, reward = env.step(state, action, goal_star)
                reward_test += reward
                # print(state_next)
                # print(goal_star)
                input_next_Q = torch.FloatTensor(
                    np.concatenate([state_next, goal_star])
                )
                if h + 1 < H:
                    out_next = qnet_HER[h + 1](input_next_Q)
                # dataset[h].append((state, action_id, state_next))
                state = state_next
        print("testing n:{}\t reward:{:.2f}\t ".format(n, reward_test / test_each))
        overall_out_HER[oit].append(reward_test / test_each)

        # generate Q
        # generate next Q
        # optimize
pd.DataFrame(np.array(overall_out_HER).T).to_csv(
    "out_HER.csv", header=False, index=False
)
