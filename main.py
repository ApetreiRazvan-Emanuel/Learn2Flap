import os
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt
from game.flappy_bird import GameState


class Config:
    MODEL_NAME = "testfinal"
    LEARNING_RATE = 1e-5  # 1e-6
    FRAME_SKIP = 1
    FRAME_SKIP_JUMP = 0
    SHOW_GAME = True
    NUMBER_OF_ACTIONS = 2
    GAMMA = 0.99
    INITIAL_EPSILON = 0.2 # 0.1
    FINAL_EPSILON = 0.00001  # 0.0001
    NUMBER_OF_ITERATIONS = 2000000
    REPLAY_MEMORY_SIZE = 10000
    MINIBATCH_SIZE = 32
    TARGET_UPDATE_FREQUENCY = 1000


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(3136, 512)
        self.fc5 = nn.Linear(512, Config.NUMBER_OF_ACTIONS)

    def forward(self, x):
        output = self.conv1(x)
        output = torch.nn.functional.relu(output)
        output = self.conv2(output)
        output = torch.nn.functional.relu(output)
        output = self.conv3(output)
        output = torch.nn.functional.relu(output)
        output = output.view(output.size()[0], -1)
        output = self.fc4(output)
        output = torch.nn.functional.relu(output)
        output = self.fc5(output)

        return output


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


i = 0


def image_processing(image):
    # image = image[40:288, 20:360] test
    image = image[:, 40:300]
    # image = image[40:288, 30:320]
    # global i
    # i += 1
    # if i % 500 == 0:
    #     plt.imshow(image)
    #     plt.show()
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    # normalized_img = image_data / 255.0
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    # global i
    # i += 1
    # if i % 500 == 0:
    #     plt.imshow(image_data, cmap='gray')
    #     plt.show()
    image_tensor = image_data.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def optimize_model(replay_memory, model, target_model, optimizer, loss_function):
    minibatch = random.sample(replay_memory, min(len(replay_memory), Config.MINIBATCH_SIZE))

    state_batch = torch.cat(tuple(d[0] for d in minibatch))
    action_batch = torch.cat(tuple(d[1] for d in minibatch))
    reward_batch = torch.cat(tuple(d[2] for d in minibatch))
    state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

    if torch.cuda.is_available():
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        state_1_batch = state_1_batch.cuda()

    output_1_batch = target_model(state_1_batch)

    y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                              else reward_batch[i] + Config.GAMMA * torch.max(output_1_batch[i])
                              for i in range(len(minibatch))))

    q_value = torch.sum(model(state_batch) * action_batch, dim=1)

    optimizer.zero_grad()

    y_batch = y_batch.detach()

    loss = loss_function(q_value, y_batch)

    loss.backward()
    optimizer.step()


def train():
    model = NeuralNetwork()
    target_model = NeuralNetwork()

    if torch.cuda.is_available():
        model = model.cuda()
        target_model = target_model.cuda()

    model.apply(init_weights)
    target_model.load_state_dict(model.state_dict())
    start = time.time()

    frames_to_skip = 0

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_function = nn.MSELoss()

    game_state = GameState(Config.SHOW_GAME)
    replay_memory = []

    action = torch.zeros([Config.NUMBER_OF_ACTIONS], dtype=torch.float32)
    action[0] = 1
    image_data, reward, finished = game_state.frame_step(action)
    image_data = image_processing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    epsilon = Config.INITIAL_EPSILON
    iteration = 0

    epsilon_decrements = np.linspace(Config.INITIAL_EPSILON, Config.FINAL_EPSILON, Config.NUMBER_OF_ITERATIONS)

    while iteration < Config.NUMBER_OF_ITERATIONS:
        output = model(state)[0]

        action = torch.zeros([Config.NUMBER_OF_ACTIONS], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        skipped_frame = False
        if frames_to_skip > 0:
            skipped_frame = True
            frames_to_skip -= 1
            action_index = [torch.tensor(0)][0]
        else:
            if random.random() <= epsilon:
                action_index = random.randint(0, Config.NUMBER_OF_ACTIONS - 1)
            else:
                action_index = torch.argmax(output).item()

        action[action_index] = 1

        if action_index == 1:
            frames_to_skip += Config.FRAME_SKIP_JUMP
        frames_to_skip += Config.FRAME_SKIP if not skipped_frame else 0

        image_data_1, reward, finished = game_state.frame_step(action)
        image_data_1 = image_processing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, finished))

        if len(replay_memory) > Config.REPLAY_MEMORY_SIZE:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]

        optimize_model(replay_memory, model, target_model, optimizer, loss_function)

        state = state_1
        iteration += 1

        if iteration % Config.TARGET_UPDATE_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())

        if iteration % 25000 == 0:
            torch.save(model, f"models/{Config.MODEL_NAME}_" + str(iteration) + ".pth")

        if iteration % 1000 == 0:
            print("Iteratia:", iteration, "timp:", time.time() - start, "epsilon:", epsilon, "actiunea:",
                  action_index.cpu().detach().numpy(), "recompensa:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()))


def test(model):
    game_state = GameState(Config.SHOW_GAME)

    action = torch.zeros([Config.NUMBER_OF_ACTIONS], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = image_processing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]

        action = torch.zeros([Config.NUMBER_OF_ACTIONS], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = image_processing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main(model_path=None):
    cuda_is_available = torch.cuda.is_available()

    if model_path:
        model = torch.load(model_path).eval()

        if cuda_is_available:
            model = model.cuda()

        test(model)

    else:
        if not os.path.exists('models/'):
            os.mkdir('models/')

        train()


if __name__ == "__main__":
    main("models/FinalFrameskip1learningrate1e5_1500000.pth")
