
from torch.utils import data
from torchvision import transforms as T
import os
import numpy as np
import SimpleITK as sitk
from utils import tsfm_tfusion
import glob


def resolve_volume_path(subject_dir, filename_stem):
    """Resolve a volume regardless of whether it is stored as .nii or .nii.gz."""
    candidates = [
        os.path.join(subject_dir, f"{filename_stem}.nii.gz"),
        os.path.join(subject_dir, f"{filename_stem}.nii"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Missing NIfTI volume. Tried: {', '.join(candidates)}"
    )

class Brain(data.Dataset):
    def __init__(self, data_file, selected_modal, inputs_transform=None,
                 labels_transform=None, t_join_transform=None, join_transform=None, phase='train'):
        self.selected_modal = selected_modal
        self.c_dim = len(self.selected_modal)
        self.inputs_transform = inputs_transform
        self.labels_transform = labels_transform
        self.join_transform = join_transform
        self.t_join_transform = t_join_transform
        self.data_file = data_file
        self.dataset = {}
        self.phase = phase
        self.init()

    def init(self):

        self.dataset['data'] = []
        lines = [line.rstrip() for line in open(self.data_file, 'r')]
        flag_m = 0
        flag_pid = "-1"
        for i, image_path in enumerate(lines):

            image_name = os.path.basename(image_path)
            pid = image_name.split('_')[-1]
            if pid != flag_pid:
                flag_pid = pid
                flag_m += 1
                flag_m %= 15

            if self.phase == 'test':
                for k in range(15):
                    self.dataset['data'].append([image_path, pid,  k + 1])
            else:
                self.dataset['data'].append([image_path, pid, flag_m % 15 + 1])

        print('[*] Load {}, which contains {} paired volume\nmodilities: {}'.format(self.data_file,
                                                                                       len(self.dataset['data']),
                                                                                       self.selected_modal))
    def __getitem__(self, idex):

        image_path, pid, m_d = self.dataset['data'][idex]
        # label_path = glob.glob(os.path.join(image_path, f'BraTS19_*_{pid}_seg.nii.gz'))
        
        # label_path = image_path + '/BraTS20_Training_{}_{}.nii.gz'.format(pid, 'seg')
        # label_path = image_path + '/BraTS19_Training_{}_{}.nii.gz'.format(pid, 'seg')
        folder_name = os.path.basename(image_path)
        label_path = resolve_volume_path(image_path, folder_name + '_seg')
        # label_path = image_path + '/{}-{}.nii'.format(pid, 'seg')


        volume_label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)

        volumes = []
        crop_size = None
        for i in range(len(self.selected_modal)):
            m_path = resolve_volume_path(
                image_path, f"{folder_name}_{self.selected_modal[i]}"
            )
            # m_path = image_path + '/BraTS20_Training_{}_{}.nii.gz'.format(pid, self.selected_modal[i])
            # m_path = image_path + '/BraTS19_Training_{}_{}.nii.gz'.format(pid, self.selected_modal[i])
            # m_path = image_path + '/{}-{}.nii.gz'.format(pid, self.selected_modal[i])

            volumes.append(sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.float32))

        if self.join_transform:
            volumes, volume_label, crop_size = self.join_transform(volumes, volume_label, self.phase)
        if self.t_join_transform:
            volumes, volume_label, _ = self.t_join_transform(volumes, volume_label, self.phase)

        if self.inputs_transform:
            for i in range(len(volumes)):
                volumes[i] = self.inputs_transform(volumes[i])

        if self.labels_transform:
            volume_label = self.labels_transform(volume_label)

        # return volumes[0], volumes[1], volumes[2], volumes[3], \
        # Trailing metadata preserves all existing tuple indices while making
        # it possible to identify the exact source record in failure logs.
        return tuple(volumes) +(volume_label, pid, m_d, crop_size, image_path)

    def __len__(self):
        return len(self.dataset['data'])

def get_loaders(data_files, selected_modals, batch_size=1, num_workers=0):
    rs = np.random.RandomState(1234)
    join_tsfm = tsfm_tfusion.Compose([
        tsfm_tfusion.ThrowFirstZ(),
        tsfm_tfusion.RandomCrop(128)
    ])
    train_join_tsfm = tsfm_tfusion.Compose([
        tsfm_tfusion.RandomFlip(rs),
        tsfm_tfusion.RandomRotate(rs, angle_spectrum=10),
    ])
    input_tsfm = T.Compose([
        tsfm_tfusion.Normalize(),
        tsfm_tfusion.NpToTensor()
    ])
    label_tsfm = T.Compose([
        tsfm_tfusion.ToLongTensor()
    ])



    datasets = dict(train=Brain(data_files['train'], selected_modals,  inputs_transform=input_tsfm,
                        labels_transform=label_tsfm, t_join_transform=train_join_tsfm, join_transform=join_tsfm, phase='train'),
                    val=Brain(data_files['val'], selected_modals, inputs_transform=input_tsfm,
                        labels_transform=label_tsfm, t_join_transform=None, join_transform=join_tsfm, phase='val'),
                    # test=Brain(data_files['test'], selected_modals, brain_dir, inputs_transform=input_tsfm,
                    #     labels_transform=label_tsfm, t_join_transform=None, join_transform=join_tsfm, phase='test')
                    )
    loader_options = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_options['prefetch_factor'] = 2

    loaders = {
        split: data.DataLoader(
            dataset=datasets[split],
            shuffle=(split == 'train'),
            **loader_options,
        )
        for split in ('train', 'val')
    }
    return loaders
