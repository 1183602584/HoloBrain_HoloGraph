import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.utils import to_dense_adj
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure


class HCPA_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_mapping = {
            "REST": 0,
            "CARIT": 1,
            "FACENAME": 2,
            "VISMOTOR": 3,
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        bold_files = [f for f in os.listdir(self.bold_dir) if f.endswith(".csv")]
        for bold_file in bold_files:
            try:
                parts = bold_file.split("_")
                subject_id = parts[0]
                task_type = parts[1].split("-")[1]
                if task_type not in self.label_mapping:
                    continue
                sc_path = os.path.join(self.sc_dir, subject_id, f"{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
                if not os.path.exists(sc_path):
                    continue
                mat = loadmat(sc_path)
                if 'aal116_radius2_count_connectivity' not in mat:
                    raise KeyError(f"AAL not found in SC file: {sc_path}")
                sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)

                bold_path = os.path.join(self.bold_dir, bold_file)
                bold_data = pd.read_csv(bold_path).values[:, 1:]
                bold_data = self.pad_sentences(bold_data) if bold_data.shape[0] < 300 else bold_data[:300]
                label = self.label_mapping[task_type]
                data.append((bold_data, sc, label))
            except Exception as e:
                print(f"Error processing file {bold_file}: {e}")
        return data

    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(300 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]
        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class TUD(Dataset):
    def __init__(self, root: str, name: str):
        self.dataset = TUDataset(root=root, name=name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = data.x.T  
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        label = data.y
        return x, adj, label


class HCP_YA_SCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_path, scan="LR"):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_path = label_path
        self.labels = pd.read_csv(label_path).values.flatten()
        self.bold_files = [f for f in os.listdir(bold_dir) if "WM" in f and scan in f]
        self.data = []
        for bold_file in self.bold_files:
            subject_id = bold_file.split("_")[0].split("-")[1]
            sc_path = os.path.join(sc_dir, f"sub-{subject_id}", f"sub-{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
            if not os.path.exists(sc_path):
                continue
            sc_dict = loadmat(sc_path)
            if 'brainnetome246_radius2_count_connectivity' not in sc_dict:
                continue
            sc_matrix = sc_dict['brainnetome246_radius2_count_connectivity']
            bold_path = os.path.join(bold_dir, bold_file)
            bold_data = pd.read_csv(bold_path).values[:, 1:]
            segments, segment_labels = self._process_bold_segments(bold_data, self.labels)
            for segment, label in zip(segments, segment_labels):
                self.data.append((segment, sc_matrix, label))

    def _process_bold_segments(self, bold_data, labels):
        segments = []
        segment_labels = []
        current_label = labels[0]
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if 1 <= current_label <= 8:
                    segments.append(bold_data[start_idx:i])
                    segment_labels.append(current_label - 1)
                start_idx = i
                current_label = labels[i]
        if 1 <= current_label <= 8:
            segments.append(bold_data[start_idx:])
            segment_labels.append(current_label - 1)
        assert len(segments) == len(segment_labels) == 8
        return segments, segment_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold_segment, sc_matrix, label = self.data[idx]
        bold_tensor = torch.tensor(bold_segment, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc_matrix, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class HCPYA_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_mapping = {
            "EMOTION": 0,
            "GAMBLING": 1,
            "LANGUAGE": 2,
            "MOTOR": 3,
            "RELATIONAL": 4,
            "SOCIAL": 5,
            "WM": 6,
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        bold_files = [f for f in os.listdir(self.bold_dir) if f.endswith(".csv") and "LR" in f]
        for bold_file in bold_files:
            try:
                parts = bold_file.split("_")
                subject_id = parts[0]
                task_type = parts[1].split("-")[1]
                if task_type not in self.label_mapping:
                    continue
                sc_path = os.path.join(self.sc_dir, subject_id, f"{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
                if not os.path.exists(sc_path):
                    continue
                mat = loadmat(sc_path)
                if 'aal116_radius2_count_connectivity' not in mat:
                    raise KeyError(f"AAL not found in SC file: {sc_path}")
                sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)
                bold_path = os.path.join(self.bold_dir, bold_file)
                bold = pd.read_csv(bold_path, header=0).values[:, 1:]
                bold = self.pad_sentences(bold) if bold.shape[0] < 175 else bold[:175]
                label = self.label_mapping[task_type]
                data.append((bold, sc, label))
            except Exception as e:
                print(f"Error processing file {bold_file}: {e}")
        return data

    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(175 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]
        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class HCPYA_byregion(Dataset):
    def __init__(self, bold_dir, label_path):
        self.bold_dir = bold_dir
        self.labels = self._load_labels(label_path)
        self.num_regions = 116
        self.data = self._load_and_cache_bold_data()

    def _load_and_cache_bold_data(self):
        bold_files = [os.path.join(self.bold_dir, f) for f in os.listdir(self.bold_dir)
                      if f.endswith(".csv") and "LR" in f]
        all_bold_data = []
        for bold_file in bold_files:
            bold = pd.read_csv(bold_file, header=0).values[:, 1:]  # [T, 116]
            bold_tensor = torch.tensor(bold, dtype=torch.float32)
            bold_tensor = self.pad_sentences(bold_tensor) if bold_tensor.shape[0] < 175 else bold_tensor[:175]
            all_bold_data.append(bold_tensor)
        return torch.stack(all_bold_data, dim=0)[:900]  # [num_subjects, 175, num_regions]

    def pad_sentences(self, sentence):
        pad_data = torch.cat((sentence, torch.zeros(175 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def _load_labels(self, label_path):
        labels = np.loadtxt(label_path)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_regions

    def __getitem__(self, region_idx):
        bold_region = self.data[:, :, region_idx] 
        bold_region = bold_region.permute(1, 0)  
        label = self.labels[region_idx] - 1
        return bold_region, bold_region, label
    

class HCPA_byregion(Dataset):
    def __init__(self, bold_dir, label_path):

        self.bold_dir = bold_dir
        self.labels = self._load_labels(label_path) 
        self.num_regions = 116                      
        self.data = self._load_and_cache_bold_data() 

    def _load_and_cache_bold_data(self):

        bold_files = [os.path.join(self.bold_dir, f) for f in os.listdir(self.bold_dir) if f.endswith(".csv") and "AP" in f and "REST" in f]
        all_bold_data = []

        for bold_file in bold_files:
            bold = pd.read_csv(bold_file, header=0).values[:, 1:]  # [T, 116]
            bold_tensor = torch.tensor(bold, dtype=torch.float32)
            bold_tensor = self.pad_sentences(bold_tensor) if bold_tensor.shape[0] < 300 else bold_tensor[:300] 
            all_bold_data.append(bold_tensor)

        return torch.stack(all_bold_data, dim=0)[:900] 
    
    def pad_sentences(self, sentence):
        pad_data = torch.cat((sentence, torch.zeros(300 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def _load_labels(self, label_path):
        labels = np.loadtxt(label_path)
        return torch.tensor(labels, dtype=torch.long) 

    def __len__(self):
        return self.num_regions

    def __getitem__(self, region_idx):

        bold_region = self.data[:, :, region_idx]  
        bold_region = bold_region.permute(1, 0)  
        label = self.labels[region_idx]-1      

        return bold_region, bold_region, label
    


import os
import re
import glob
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class ABIDE_BoldFCDataset(Dataset):
    """
    返回: (features, adj, label)
    features: (T, N) float32  (BOLD ROI time series)
    adj:      (N, N) float32  (FC as adjacency)
    label:    int64           (0/1)  ASD=1, Control=0
    """

    def __init__(
        self,
        ts_dir="/home/hezhenkun/nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal",
        phenotypic_csv="/home/hezhenkun/nilearn_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
        atlas="aal",
        expected_n=116,
        fix_len=150,
        k=None,
        use_abs=True,
        zscore=True,
        cache_fc_path=None,   # 例如 "/home/hezhenkun/nilearn_data/abide_fc_cache.npy"
    ):
        super().__init__()
        self.fix_len = int(fix_len)
        self.k = k
        self.use_abs = use_abs
        self.zscore = zscore
        self.expected_n = int(expected_n)

        ts_dir = os.path.expanduser(ts_dir)
        phenotypic_csv = os.path.expanduser(phenotypic_csv)

        # 1) 找到所有 rois_{atlas}.1D 文件
        pattern = os.path.join(ts_dir, f"*rois_{atlas}.1D")
        self.files = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise FileNotFoundError(f"没有找到任何文件：{pattern}")

        # 2) 读取 phenotypic，建立 FILE_ID -> label 的映射
        fileid_to_label = self._load_labels_from_csv(phenotypic_csv)

        cleaned = []
        labels = []
        used_files = []

        for f in self.files:
            file_id = self._infer_file_id_from_filename(f)  # e.g., "Pitt_0050003"
            if file_id not in fileid_to_label:
                # phenotypic 里可能没有这条（或命名不一致），直接跳过
                continue

            X = self._load_1d_timeseries(f)  # (T, N) float32
            if X is None:
                continue

            # 对齐 N=116（AAL）
            X = self._ensure_shape_TN(X, expected_n=self.expected_n)
            if X is None:
                continue

            if X.shape[0] < self.fix_len:
                continue
            X = X[: self.fix_len].astype(np.float32)

            # z-score per ROI（按列）
            if self.zscore:
                mu = X.mean(axis=0, keepdims=True)
                sd = X.std(axis=0, keepdims=True) + 1e-8
                X = (X - mu) / sd

            cleaned.append(X)
            labels.append(fileid_to_label[file_id])
            used_files.append(f)

        if len(cleaned) == 0:
            raise RuntimeError(
                "清洗后样本为 0。请检查：\n"
                f"1) ts_dir 是否正确：{ts_dir}\n"
                f"2) phenotypic_csv 是否正确：{phenotypic_csv}\n"
                f"3) 文件名 FILE_ID 与 CSV 的 FILE_ID 是否能匹配（如 Pitt_0050003）\n"
            )

        self.cleaned = cleaned
        self.labels = np.array(labels, dtype=np.int64)
        self.used_files = used_files  # 方便你debug

        # 3) 计算 FC 当邻接（一次性算好）
        if cache_fc_path is not None and os.path.exists(cache_fc_path):
            fc = np.load(cache_fc_path).astype(np.float32)
            if fc.shape[0] != len(self.cleaned):
                raise ValueError(f"cache_fc_path 样本数不匹配：cache={fc.shape[0]} vs cleaned={len(self.cleaned)}")
        else:
            fc = self._compute_fc_all(self.cleaned)  # (S, N, N)
            if cache_fc_path is not None:
                os.makedirs(os.path.dirname(cache_fc_path), exist_ok=True)
                np.save(cache_fc_path, fc)

        fc = np.nan_to_num(fc).astype(np.float32)
        if self.use_abs:
            fc = np.abs(fc)

        if self.k is not None:
            fc = self._topk_sparsify(fc, k=int(self.k))

        self.fc = fc

    # ------------------ IO & parsing ------------------

    def _load_labels_from_csv(self, phenotypic_csv):
        """
        CSV 里常见列：FILE_ID, DX_GROUP
        DX_GROUP: 1=ASD, 2=Control
        输出：{FILE_ID: label}，label: ASD=1, Control=0
        """
        if not os.path.exists(phenotypic_csv):
            raise FileNotFoundError(f"找不到 phenotypic CSV：{phenotypic_csv}")

        fileid_to_label = {}
        with open(phenotypic_csv, "r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            # 兼容不同大小写/字段名
            headers = reader.fieldnames or []
            def pick_col(cands):
                for c in cands:
                    if c in headers:
                        return c
                # 再试试忽略大小写
                low = {h.lower(): h for h in headers}
                for c in cands:
                    if c.lower() in low:
                        return low[c.lower()]
                return None

            col_file = pick_col(["FILE_ID", "file_id", "SUB_ID", "subject_id"])
            col_dx = pick_col(["DX_GROUP", "dx_group", "DX", "diagnosis"])

            if col_file is None or col_dx is None:
                raise ValueError(f"CSV 缺少必要列。已有列：{headers}，需要 FILE_ID 与 DX_GROUP")

            for row in reader:
                fid = str(row[col_file]).strip()
                dx = row[col_dx]
                try:
                    dx = int(float(dx))
                except Exception:
                    continue

                # DX_GROUP: 1=ASD, 2=Control
                label = 1 if dx == 1 else 0
                fileid_to_label[fid] = label

        return fileid_to_label

    def _infer_file_id_from_filename(self, path):
        """
        例如：".../Pitt_0050003_rois_aal.1D" -> "Pitt_0050003"
        """
        base = os.path.basename(path)
        m = re.match(r"(.+?)_rois_[A-Za-z0-9]+\.1D$", base)
        if m:
            return m.group(1)
        # 兜底：去掉后缀
        return base.replace(".1D", "")

    def _load_1d_timeseries(self, path):
        """
        读取 .1D：通常是空格/制表符分隔的纯数值矩阵
        返回 numpy (T, N) 或 None
        """
        try:
            X = np.loadtxt(path, dtype=np.float32)
        except Exception:
            return None

        if X.ndim == 1:
            # 单行/单列也算失败
            return None
        return X

    def _ensure_shape_TN(self, X, expected_n=116):
        """
        让数据变成 (T, N=expected_n)
        - 如果是 (N, T) 就转置
        - 如果列数 > expected_n，取前 expected_n 列（或你可以改成取后 expected_n 列）
        - 如果列数 < expected_n，返回 None（说明不是 AAL 116）
        """
        T, N = X.shape[0], X.shape[1]

        # 如果看起来像 (N,T)，就转
        if T == expected_n and N != expected_n:
            X = X.T
            T, N = X.shape

        if N < expected_n:
            return None
        if N > expected_n:
            X = X[:, :expected_n]

        return X

    # ------------------ FC computing ------------------

    def _compute_fc_all(self, cleaned_list):
        """
        cleaned_list: list of (T,N)
        return (S,N,N)
        """
        fc_all = []
        for X in cleaned_list:
            # corrcoef: rowvar=False -> 每列是变量(ROI)，每行是时间点
            C = np.corrcoef(X, rowvar=False).astype(np.float32)  # (N,N)
            C = np.nan_to_num(C)
            fc_all.append(C)
        return np.stack(fc_all, axis=0)

    def _topk_sparsify(self, fc_all, k=15):
        out_all = []
        for fc in fc_all:
            N = fc.shape[0]
            W = fc.copy()
            np.fill_diagonal(W, 0.0)
            out = np.zeros_like(W, dtype=np.float32)

            # 每行取 top-k
            idx = np.argpartition(W, -k, axis=1)[:, -k:]
            rows = np.arange(N)[:, None]
            out[rows, idx] = W[rows, idx]

            # 对称化
            out = 0.5 * (out + out.T)
            out_all.append(out.astype(np.float32))
        return np.stack(out_all, axis=0)

    # ------------------ torch Dataset ------------------

    def __len__(self):
        return len(self.cleaned)

    def __getitem__(self, idx):
        features = torch.tensor(self.cleaned[idx], dtype=torch.float32)  # (T,N)
        adj = torch.tensor(self.fc[idx], dtype=torch.float32)            # (N,N)
        label = torch.tensor(self.labels[idx], dtype=torch.long)         # scalar
        return features, adj, label