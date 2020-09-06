import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from matplotlib import pyplot

from torch import nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cv2

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        # self.counter_num = 0
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, A, counter_num = 0):
        if self.display_id > 0:  # show images in the browser
            # self.opt.display_single_pane_ncols = A
            ncols = self.opt.display_single_pane_ncols
            
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            subpath = "/" + str(epoch) + "/"
            subfoler_path = self.img_dir + subpath
            if not os.path.exists(subfoler_path):
                os.makedirs(subfoler_path)
            for label, image_numpy in visuals.items():
                if A == 1:
                    # print("save_result :", save_result)
                    img_path = os.path.join(subfoler_path, 'Valid_epoch%.3d_%s_%s.png' % (epoch, label, str(counter_num)))
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                
                for label, image_numpy in visuals.items():
                    if A == 1:

                        img_path = 'Valid_epoch%.3d_%s.png' % (n, label)
                    else:
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):

        self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        print("Erros :", errors)
        pyplot['X'].append(epoch + counter_ratio)
        pyplot['Y'].append([errors[k] for k in self.plot_data['legend']])
        pyplot.savefig(opt.checkpoints_dir + '/plot_line_plot_loss.png')
        pyplot.close()
        
    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, A):
        k_total = []
        v_total = []
        if A == 1:
            message = '(Test epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
            for k, v in errors.items():
                # print("k :", k)
                # print("v :", v)
                message += '%s: %.3f ' % (k, v)
                k_total.append(k)
                v_total.append(v)
            print(message)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        else:
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
            for k, v in errors.items():
                # print("k :", k)
                # print("v :", v)
                message += '%s: %.3f ' % (k, v)
                k_total.append(k)
                v_total.append(v)
            print(message)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        return k_total, v_total

        
    def save_valid_images(self,  visuals, image_path, epoch, counter_man):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        subpath = "/" + str(epoch) + "/"
        subfoler_path = self.img_dir + subpath
        if not os.path.exists(subfoler_path):
            os.makedirs(subfoler_path)
        for label, image_numpy in visuals.items():
            image_name = '%s_%s_%s.jpg' % (image_path[0], label, str(counter_man))
            save_path = os.path.join(subfoler_path, image_name)
            print(save_path)
            util.save_image(image_numpy, save_path)

    def get_colors(self,inp, colormap=plt.cm.jet, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

    def show_attention(self,featmap):

        featmap = (featmap[0]/(0.00000001 + featmap[0].max(1)[0].unsqueeze(1).unsqueeze(1))).mean(0)
        featmap = cv2.resize(featmap.cpu().float().numpy(), (128, 128))
        featmap = self.get_colors(featmap, plt.cm.jet)
        featmap = featmap[:,:,:3][:,:,::-1] * 0.5 * 255 
        return featmap.astype(np.uint8)
    
    def out_im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, before_list, after_list, p1, p2):
        image_dir = webpage.get_image_dir()
        image_dir_att = image_dir[:image_dir.rindex("/")]
        image_dir_att = image_dir_att + "/Att/"
        if not os.path.exists(image_dir_att):
            os.makedirs(image_dir_att)
        print(image_dir_att)
        ims = []
        txts = []
        links = []
        counter_man = 0
        before_out = []
        after_out = []

        white = np.zeros([128, 10, 3], np.uint8)+255

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (image_path[counter_man], label)

            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            before = before_list[0][counter_man]
            after = after_list[0][counter_man]

            before_0 = self.show_attention(before)

            p1_out = p1[counter_man]
            p1_out = self.out_im(p1_out)
            p1_out = np.hstack((image_numpy, white))
            before_0 = np.hstack((image_numpy, before_0))

            after_0 = self.show_attention(after)

            p1_out = p1[counter_man]
            p1_out =(np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            p1_out = np.hstack((image_numpy, white))
            after_0 = np.hstack((image_numpy, after_0))


            for i in range(1,9):

                before = before_list[i][counter_man]
                after = after_list[i][counter_man]

                before = self.show_attention(before)
                after = self.show_attention(after)

                before_0 = np.hstack((before_0, white))
                after_0 = np.hstack((after_0, white))

                before_0 = np.hstack((before_0, before))
                after_0 = np.hstack((after_0, after))

            image_name = 'Before_Att_%s_%s.jpg' % (image_path[counter_man], label)
            save_path = os.path.join(image_dir_att, image_name)
            
            util.save_image(before_0, save_path)

            image_name = 'After_Att_%s_%s.jpg' % (image_path[counter_man], label)
            save_path = os.path.join(image_dir_att, image_name)      
            util.save_image(after_0, save_path)

            print("save_path :", save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
            counter_man = counter_man + 1

        webpage.add_images(ims, txts, links, width=self.win_size)
