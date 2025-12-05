import torch
from load_data import data_loader
import torch.utils.data as data
import torch.nn as nn
from gan_model import Generator, Discriminator
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# class Trainer(object):
#     def __init__(self, opt):

#         self.opt = opt

#         self.generator = data.DataLoader(data_loader(opt), batch_size=opt['batch_size'], shuffle=True)

#         # model
#         self.G = Generator(opt)
#         self.D = Discriminator(opt)

#         # loss function
#         self.G_loss = nn.MSELoss()
#         self.D_loss = nn.BCELoss()

#         if opt['cuda']:
#             self.G = self.G.cuda()
#             self.D = self.D.cuda()
#             self.G_loss = self.G_loss.cuda()
#             self.D_loss = self.D_loss.cuda()

#         # Optimizer
#         self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt['lr'])
#         self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt['lr'] / 10)

#     def train(self):
    
#         self.G.train()
#         self.D.train()
#         for e in range(1, self.opt['epoch']+1):
#             for step, ((recent_data, trend_data, time_feature), sub_graph, real_data, _, _) in enumerate(self.generator):
#                 """
#                 recent_data: (batch_size, time, node_num, num_feature)
#                 trend_data: (batch_size, time, num_feature)
#                 real_data: (batch_size, num_adj, num_feature)
#                 """

#                 valid = torch.zeros((real_data.shape[0], 1), dtype=torch.float)
#                 fake = torch.ones((real_data.shape[0], 1), dtype=torch.float)

#                 if self.opt['cuda']:
#                     recent_data, trend_data, real_data, sub_graph, time_feature, valid, fake = \
#                         recent_data.cuda(), trend_data.cuda(), real_data.cuda(), sub_graph.cuda(), time_feature.cuda(), valid.cuda(), fake.cuda()

#                 # ---------------------
#                 #  Train Discriminator
#                 # ---------------------
#                 self.D_optim.zero_grad()
#                 real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)  # (batch_size, time, num_adj, input_size)
#                 fake_data = self.G(recent_data, trend_data, sub_graph, time_feature)

#                 fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

#                 real_score_D = self.D(real_sequence, sub_graph, trend_data)
#                 fake_score_D = self.D(fake_sequence, sub_graph, trend_data)

#                 real_loss = self.D_loss(real_score_D, valid)
#                 fake_loss = self.D_loss(fake_score_D, fake)
#                 D_total = (real_loss + fake_loss) / 2

#                 D_total.backward(retain_graph=True)
#                 self.D_optim.step()

#                 # -----------------
#                 #  Train Generator
#                 # -----------------
#                 self.G_optim.zero_grad()
#                 fake_data = self.G(recent_data, trend_data, sub_graph, time_feature)

#                 mse_loss = self.G_loss(fake_data, real_data)
#                 fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)
                        
#                 fake_score = self.D(fake_sequence, sub_graph, trend_data)

#                 binary_loss = self.D_loss(fake_score, valid)
#                 G_total = self.opt['lambda_G'] * mse_loss + binary_loss

#                 G_total.backward()
#                 self.G_optim.step()
                
#                 if step % 100 == 0:
#                     count = 0
#                     for score in real_score_D:
#                         if torch.mean(score) < 0.5:
#                             count += 1
#                     for score in fake_score_D:
#                         if torch.mean(score) > 0.5:
#                             count += 1

#                     acc = count / (self.opt['batch_size'] * 2)
#                     logging.info("epoch:%d step:%d [D loss: %f D acc: %.2f] [G mse: %f G binary %f]" % (e, step, D_total.detach().cpu().item(), acc * 100, mse_loss, binary_loss))

#             torch.save(self.G, self.opt['save_path'] + 'G_' + str(e) + '.pth')
#             torch.save(self.D, self.opt['save_path'] + 'D_' + str(e) + '.pth')

# class Trainer(object):
#     def __init__(self, opt):

#         self.opt = opt

#         self.generator = data.DataLoader(data_loader(opt),
#                                          batch_size=opt['batch_size'],
#                                          shuffle=True)

#         # model
#         self.G = Generator(opt)
#         self.D = Discriminator(opt)

#         # loss function
#         self.G_loss = nn.MSELoss()
#         self.D_loss = nn.BCELoss()

#         if opt['cuda']:
#             self.G = self.G.cuda()
#             self.D = self.D.cuda()
#             self.G_loss = self.G_loss.cuda()
#             self.D_loss = self.D_loss.cuda()

#         # Optimizer
#         self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt['lr'])
#         self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt['lr'] / 10)

#     def train(self):

#         self.G.train()
#         self.D.train()

#         for e in range(1, self.opt['epoch'] + 1):
#             for step, ((recent_data, trend_data, time_feature),
#                        sub_graph, real_data, _, _) in enumerate(self.generator):

#                 """
#                 recent_data: (B, T, num_adj, num_feature)
#                 trend_data:  (B, T, num_feature)
#                 real_data:   (B, num_adj, num_feature)
#                 """

#                 B = real_data.shape[0]

#                 # Labels: real = 1, fake = 0
#                 valid = torch.ones((B, 1), dtype=torch.float)
#                 fake = torch.zeros((B, 1), dtype=torch.float)

#                 if self.opt['cuda']:
#                     recent_data = recent_data.cuda()
#                     trend_data = trend_data.cuda()
#                     real_data = real_data.cuda()
#                     sub_graph = sub_graph.cuda()
#                     time_feature = time_feature.cuda()
#                     valid = valid.cuda()
#                     fake = fake.cuda()

#                 # -------------------------------------------------
#                 #  Train Discriminator  (on final frame only)
#                 # -------------------------------------------------
#                 self.D_optim.zero_grad()

#                 # real final frame
#                 real_frame = real_data  # (B, num_adj, num_feature)

#                 # generated fake frame
#                 fake_frame = self.G(recent_data, trend_data, sub_graph, time_feature).detach()

#                 # Discriminator predictions
#                 real_score = self.D(real_frame)
#                 fake_score = self.D(fake_frame)

#                 real_loss = self.D_loss(real_score, valid)
#                 fake_loss = self.D_loss(fake_score, fake)
#                 D_total = (real_loss + fake_loss) / 2

#                 D_total.backward()
#                 self.D_optim.step()

#                 # -------------------------------------------------
#                 #  Train Generator
#                 # -------------------------------------------------
#                 self.G_optim.zero_grad()

#                 fake_frame = self.G(recent_data, trend_data, sub_graph, time_feature)

#                 # Reconstruction loss
#                 mse_loss = self.G_loss(fake_frame, real_data)

#                 # Adversarial loss — want fake → real
#                 fake_score_for_G = self.D(fake_frame)
#                 binary_loss = self.D_loss(fake_score_for_G, valid)

#                 G_total = self.opt['lambda_G'] * mse_loss + binary_loss

#                 G_total.backward()
#                 self.G_optim.step()

#                 # ------------------------
#                 # Logging
#                 # ------------------------
#                 if step % 100 == 0:

#                     # D accuracy
#                     acc_real = (real_score > 0.5).float().mean()
#                     acc_fake = (fake_score < 0.5).float().mean()
#                     D_acc = 0.5 * (acc_real + acc_fake)

#                     logging.info(
#                         "epoch:%d step:%d [D loss: %.6f D acc: %.2f] "
#                         "[G mse: %.6f G adv: %.6f]" %
#                         (e, step,
#                          D_total.item(),
#                          D_acc.item() * 100,
#                          mse_loss.item(),
#                          binary_loss.item())
#                     )

#             torch.save(self.G, self.opt['save_path'] + f'G_{e}.pth')
#             torch.save(self.D, self.opt['save_path'] + f'D_{e}.pth')

class Trainer(object):
    def __init__(self, opt):

        self.opt = opt

        self.generator = data.DataLoader(data_loader(opt),
                                         batch_size=opt['batch_size'],
                                         shuffle=True)

        # model
        self.G = Generator(opt)
        self.D = Discriminator(opt)

        # loss function
        self.G_loss = nn.MSELoss()
        self.D_loss = nn.BCELoss()

        if opt['cuda']:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.G_loss = self.G_loss.cuda()
            self.D_loss = self.D_loss.cuda()

        # Optimizer
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt['lr'])
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt['lr'] / 10)

    def train(self):

        self.G.train()
        self.D.train()

        for e in range(1, self.opt['epoch'] + 1):
            for step, ((recent_data, trend_data, time_feature),
                       sub_graph, real_data, _, _) in enumerate(self.generator):

                """
                recent_data: (B, T, num_adj, num_feature)
                trend_data:  (B, T, num_feature)
                real_data:   (B, num_adj, num_feature)
                """

                B = real_data.shape[0]

                # Labels: real = 1, fake = 0
                valid = torch.ones((B, 1), dtype=torch.float)
                fake = torch.zeros((B, 1), dtype=torch.float)

                if self.opt['cuda']:
                    recent_data = recent_data.cuda()
                    trend_data = trend_data.cuda()
                    real_data = real_data.cuda()
                    sub_graph = sub_graph.cuda()
                    time_feature = time_feature.cuda()
                    valid = valid.cuda()
                    fake = fake.cuda()

                # -------------------------------------------------
                #  Train Discriminator  (on final frame only)
                # -------------------------------------------------
                if step % 5 == 0:
                    self.D_optim.zero_grad()

                    # real final frame
                    real_frame = real_data  # (B, num_adj, num_feature)

                    # generated fake frame
                    fake_frame = self.G(recent_data, trend_data, sub_graph, time_feature).detach()

                    # Discriminator predictions
                    real_score = self.D(real_frame)
                    fake_score = self.D(fake_frame)

                    real_loss = self.D_loss(real_score, valid)
                    fake_loss = self.D_loss(fake_score, fake)
                    D_total = (real_loss + fake_loss) / 2

                    D_total.backward()
                    self.D_optim.step()

                # -------------------------------------------------
                #  Train Generator
                # -------------------------------------------------
                self.G_optim.zero_grad()

                fake_frame = self.G(recent_data, trend_data, sub_graph, time_feature)

                # Reconstruction loss
                mse_loss = self.G_loss(fake_frame, real_data)

                # Adversarial loss — want fake → real
                fake_score_for_G = self.D(fake_frame)
                binary_loss = self.D_loss(fake_score_for_G, valid)

                G_total = self.opt['lambda_G'] * mse_loss + binary_loss

                G_total.backward()
                self.G_optim.step()

                # ------------------------
                # Logging
                # ------------------------
                if step % 100 == 0:

                    # D accuracy
                    acc_real = (real_score > 0.5).float().mean()
                    acc_fake = (fake_score < 0.5).float().mean()
                    D_acc = 0.5 * (acc_real + acc_fake)

                    logging.info(
                        "epoch:%d step:%d [D loss: %.6f D acc: %.2f] "
                        "[G mse: %.6f G adv: %.6f]" %
                        (e, step,
                         D_total.item(),
                         D_acc.item() * 100,
                         mse_loss.item(),
                         binary_loss.item())
                    )

                if step % 1000 == 0:
                    torch.save(self.G, self.opt['save_path'] + f'G_{step}_{e}.pth')
                    torch.save(self.D, self.opt['save_path'] + f'D_{step}_{e}.pth')

            torch.save(self.G, self.opt['save_path'] + f'G_{e}.pth')
            torch.save(self.D, self.opt['save_path'] + f'D_{e}.pth')