from gyraudio.audio_separation.data.remixed import RemixedAudioDataset
from torch import rand, randint


class RemixedRandomAudioDataset(RemixedAudioDataset):
    def get_idx_noise(self, idx):
        return randint(0, len(self.file_list)-1, (1,))

    def get_snr(self, idx):
        if self.snr_filter is None:
            return self.min_snr + (self.max_snr - self.min_snr)*rand(1)
        return self.snr_filter[randint(0, len(self.snr_filter), (1,))]
