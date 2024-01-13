from gyraudio.audio_separation.data.remixed import RemixedAudioDataset
import torch

class RemixedFixedAudioDataset(RemixedAudioDataset):
    def generate_snr_list(self) :
        rnd_gen = torch.Generator()
        rnd_gen.manual_seed(2147483647)
        if self.snr_filter is None :
            self.snr_list = self.min_snr + (self.max_snr - self.min_snr)*torch.rand(len(self.file_list), generator = rnd_gen)
        else :
            self.snr_list = self.snr_filter[torch.randint(len(self.snr_filter), len(self.file_list), generator = rnd_gen)]

    def get_idx_noise(self, idx) :
        return idx

    def get_snr(self, idx) :
        return self.snr_list[idx]