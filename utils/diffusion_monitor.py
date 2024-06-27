import torch


class LossSlidingWindow:
    def __init__(self, max_size=1000) -> None:
        self.window_max_size = max_size
        self.data_slot = []

    def push(self, data):
        if len(self.data_slot) > self.window_max_size:
            self.data_slot = self.data_slot[1:]
        self.data_slot.append(data)


class DiffusionMonitor:
    def __init__(self) -> None:
        self.scheduler = None
        self.timestep_groups = 10
        self.snr_groups = 10
        self.gamma_groups = 10
        self.loss_bin_dict = {}
        self.loss_data_window = LossSlidingWindow(max_size=1000)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.calculate_stats_x()

    def calculate_stats_x(self):
        self.gamma = self.scheduler.alphas_cumprod
        self.snr = self.gamma / (1 - self.gamma + 1e-12)
        self.timesteps = [i + 1 for i in range(len(self.gamma))]

    def update_loss(self, loss_dict):
        mse_loss_batch_mat = loss_dict["mse_loss_mat"]
        loss_mask = loss_dict["loss_mask"]
        timestep = loss_dict["timestep"]
        gamma = loss_dict["gammas"]

        bs = mse_loss_batch_mat.shape[0]
        for i in range(bs):
            loss_instance = mse_loss_batch_mat[i][loss_mask[i]].mean()
            t = timestep[i]
            self.loss_data_window.push(
                dict(timestep=t, loss=loss_instance, gamma=gamma[i])
            )

    def gen_segments(self, x_min, x_max, box_num):
        segs = []
        interval = (x_max - x_min) / box_num
        st = x_min
        for i in range(box_num):
            seg = [st, st + interval]
            segs += [seg]
            st += interval
        return segs

    def get_bins(self, x, x_segments, y):
        seg_num = len(x_segments)
        boxes = []
        for i in range(seg_num):
            boxes.append([])
        for x_, y_ in zip(x, y):
            for seg_inx, seg in enumerate(x_segments):
                if seg[0] <= x_ < seg[1]:
                    boxes[seg_inx] += [y_]
        return boxes

    def get_gamma_loss_dist(self):
        # gen data boxes
        gamma_segs = [[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]
        loss_data_dict = self.loss_data_window.data_slot
        loss_value = [i["loss"] for i in loss_data_dict]
        time_value = [i["timestep"] for i in loss_data_dict]
        gamma_value = [i["gamma"] for i in loss_data_dict]

        gamma_bins = self.get_bins(gamma_value, gamma_segs, loss_value)
        bin_average = []
        for bin in gamma_bins:
            if len(bin):
                aver = sum(bin) / len(bin)
            else:
                aver = torch.tensor(-1.0)
            bin_average += [aver]
        return gamma_segs, bin_average
