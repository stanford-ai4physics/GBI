from typing import Optional, Tuple, List, Union
from itertools import repeat
import os
import sys
import glob
import json
import subprocess

import numpy as np
import pandas as pd
import awkward as ak

from aliad.interface.awkward import Momentum4DArrayBuilder
from quickstats import stdout
from quickstats.utils.string_utils import split_str
from quickstats.maths.numerics import cartesian_product

from paws.settings import (
    NUM_SHARDS, DEFAULT_DATADIR, Sample, SampleURLs,
    MASS_RANGE, MASS_INTERVAL, BASE_SEED,
    DEFAULT_FEATURE_LEVEL
)
from paws import PathManager

def download_file(url:str, outdir:str):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = os.path.join(outdir, os.path.basename(url))
    command = ['wget', '-O', outpath, url]
    
    try:
        subprocess.run(command, check=True)
        stdout.info(f"File downloaded to {outpath}")
    except subprocess.CalledProcessError as e:
        stdout.error(f"An error occurred: {e}")

def _get_unit_scale(unit:str='GeV'):
    if unit == 'GeV':
        return 1.0
    elif unit == 'TeV':
        return 0.001
    raise ValueError(f'invalid unit: {scale} (choose between "GeV" and "TeV")')

def get_flattened_arrays(arrays:"awkward.Array", pad_size:int=200):
    jet_keys = arrays.fields
    jet_arrays = {}
    part_arrays = {}
    array_builder = Momentum4DArrayBuilder('PtEtaPhiM')
    for jet_key in jet_keys:
        if jet_key ==  'jj':
            continue
        jet_p4 = array_builder.get_array_from_dict({
            'pt' : arrays[jet_key]['jet_pt'],
            'eta': arrays[jet_key]['jet_eta'],
            'phi': arrays[jet_key]['jet_phi'],
            'm'  : arrays[jet_key]['jet_m']
        })
        jet_arrays_i = {
            f'px{jet_key}' : jet_p4.px,
            f'py{jet_key}' : jet_p4.py,
            f'pz{jet_key}' : jet_p4.pz
        }
        part_arrays_i = {}
        for field in arrays[jet_key].fields:
            array = arrays[jet_key][field]
            # particle features
            if array.ndim == 2:
                padded_arrays = ak.to_numpy(ak.fill_none(ak.pad_none(array, pad_size, clip=True), 0)).T
                for i, padded_array in enumerate(padded_arrays):
                    key = field.replace('part_', '') + f'{jet_key}p{i + 1}'
                    part_arrays_i[key] = padded_array
            # jet features
            else:
                key = field.replace('jet_', '') + jet_key
                jet_arrays_i[key] = array
        for key in jet_arrays_i:
            jet_arrays_i[key] = ak.to_numpy(jet_arrays_i[key])
        #for key in part_arrays_i:
        #    part_arrays_i[key] = ak.to_numpy(part_arrays_i[key])
        jet_arrays.update(jet_arrays_i)
        part_arrays.update(part_arrays_i)
    if 'jj' in jet_keys:
        for field in arrays['jj'].fields:
            key = field.replace('dijet_', '') + 'jj'
            jet_arrays[key] = ak.to_numpy(arrays['jj'][field])
    return {**jet_arrays, **part_arrays}

def RnD_txt_to_arrays(filename:str, sortby:Optional[str]='mass', unit:str='GeV',
                      feature_level:str='low_level', extra_vars:bool=True,
                      flatten:bool=False, pad_size:int=200):
    assert sortby in [None, 'mass', 'pt']
    assert unit in ['GeV', 'TeV']
    assert feature_level in ['low_level', 'high_level']
    stdout.info(f'Reading file "{filename}"')
    event_numbers = []
    jet_indices = []
    jet_features = []
    jet_N = []
    part_features = []
    with open(filename, 'r') as f:
        data = f.readlines()
    low_level = feature_level == 'low_level'
    scale = _get_unit_scale(unit)
    ntaus = 5 if extra_vars else 3
    for line in data:
        tokens = line.split()
        event_number = int(tokens[0])
        jet_index = int(tokens[1])
        jet_features_i = [float(token) for token in tokens[3: 13]]
        event_numbers.append(event_number)
        jet_indices.append(jet_index)
        part_tokens = line.split('P')[1:]
        jet_N = len(part_tokens)
        jet_features_i.append(jet_N)
        jet_features.append(jet_features_i)
        if low_level:
            # the last element of the tuple is the constituent mass (= 0)
            part_features_i = [tuple(split_str(tokens, cast=float) + [0.]) \
                               for tokens in part_tokens]
            part_features_i = np.array(part_features_i, dtype=[("pT", "float64"),
                                                               ("eta", "float64"),
                                                               ("phi", "float64"),
                                                               ("mass", "float64")])
            part_features.append(part_features_i)
    event_numbers = np.array(event_numbers)
    jet_indices = np.array(jet_indices)
    jet_size = np.unique(jet_indices).shape[0]
    jet_features = np.array(jet_features)
    record = {
        'event_number'  : event_numbers,
        'jet_index'     : jet_indices,
        'jet_features'  : jet_features
    }
    if low_level:
        array_builder = Momentum4DArrayBuilder('PtEtaPhiM')
        part_features = array_builder.get_array_from_list(part_features)
        record['part_features'] = part_features
    if sortby is not None:
        feature_idx_map = {
            'pt': 0,
            'mass': 3
        }
        feature_idx = feature_idx_map[sortby]
        feature_size = jet_features.shape[0]
        sort_idx = np.argsort(-jet_features[:, feature_idx].reshape(feature_size // jet_size, jet_size), axis=-1)
        sort_idx = sort_idx + np.arange(feature_size//jet_size).reshape(feature_size // jet_size, 1) * jet_size
        sort_idx = sort_idx.flatten()
        for key in ['jet_features', 'part_features']:
            if key not in record:
                continue
            record[key] = record[key][sort_idx]
    record = ak.Record(record)
    arrays = {}
    for jet_index in range(jet_size):
        jet_key = f'j{jet_index + 1}'
        jet_mask = record['jet_index'] == jet_index
        jet_features = record['jet_features'][jet_mask]
        arrays[jet_key] = {}
        # jet features
        arrays[jet_key]["jet_pt"]   = jet_features[:, 0] * scale
        arrays[jet_key]["jet_eta"]  = jet_features[:, 1]
        arrays[jet_key]["jet_phi"]  = jet_features[:, 2]
        arrays[jet_key]["jet_m"]    = jet_features[:, 3] * scale
        #arrays[jet_key]["N"]        = ak.num(part_features, axis=-1)
        arrays[jet_key]["N"]        = jet_features[:, -1]
        if extra_vars:
            arrays[jet_key]["jet_btag"] = jet_features[:, 4]
        for i in range(ntaus):
            arrays[jet_key][f'tau{i + 1}'] = jet_features[:, 5 + i]
        if extra_vars:
            tau_mask = np.array(((jet_features[:, 5] > 0) & (jet_features[:, 6] > 0)))
            arrays[jet_key]['tau12'] = np.where(tau_mask, np.divide(jet_features[:, 6],
                                                                    jet_features[:, 5], where=tau_mask), 0)
            arrays[jet_key]['tau23'] = np.where(tau_mask, np.divide(jet_features[:, 7],
                                                                    jet_features[:, 6], where=tau_mask), 0)
        jet_p4 = Momentum4DArrayBuilder.get_array_from_dict({
            'pt'  : jet_features[:, 0],
            'eta' : jet_features[:, 1],
            'phi' : jet_features[:, 2],
            'm'   : jet_features[:, 3]})
        # particle features
        if low_level:
            part_features = record['part_features'][jet_mask]
            arrays[jet_key]['part_pt']   = part_features.pt * scale
            arrays[jet_key]['part_eta']  = part_features.eta
            arrays[jet_key]['part_phi']  = part_features.phi
            if extra_vars:
                arrays[jet_key]['part_e']        = part_features.e * scale
                arrays[jet_key]['part_relpt']    = part_features.pt / jet_p4.pt
                arrays[jet_key]['part_deltaeta'] = part_features.deltaeta(jet_p4)
                arrays[jet_key]['part_deltaphi'] = part_features.deltaphi(jet_p4)
                arrays[jet_key]['part_deltaR']   = part_features.deltaR(jet_p4)
    ak_arrays = ak.Array(arrays)
    # dijet features
    if extra_vars and (jet_size == 2):
        # compute mjj
        j1_p4 = ak.zip({
        "pt"  : ak_arrays["j1"]["jet_pt"],
        "eta" : ak_arrays["j1"]["jet_eta"],
        "phi" : ak_arrays["j1"]["jet_phi"],
        "m"   : ak_arrays["j1"]["jet_m"]
        }, with_name="Momentum4D")
        j2_p4 = ak.zip({
            "pt"  : ak_arrays["j2"]["jet_pt"],
            "eta" : ak_arrays["j2"]["jet_eta"],
            "phi" : ak_arrays["j2"]["jet_phi"],
            "m"   : ak_arrays["j2"]["jet_m"]
        }, with_name="Momentum4D")
        jj_p4 = j1_p4.add(j2_p4)
        ak_arrays['jj'] = ak.Array({'dijet_m': jj_p4.m,
                                    'dijet_pt': jj_p4.pt})
    if flatten:
        return get_flattened_arrays(ak_arrays, pad_size=pad_size)
    return ak_arrays

def RnD_txt_to_parquet(filename:str, outname:str, cache:bool=True,
                       sortby:Optional[str]='mass', unit:str='GeV',
                       feature_level:str='low_level', extra_vars:bool=True,
                       pad_size:int=200):
    if cache and (outname is not None) and os.path.exists(outname):
        stdout.info(f'Cached output from "{outname}"')
        return 
    arrays = RnD_txt_to_arrays(filename, sortby=sortby, unit=unit,
                               feature_level=feature_level,
                               extra_vars=extra_vars,
                               flatten=False,
                               pad_size=pad_size)
    stdout.info(f'Saving output to "{outname}"')
    ak.to_parquet(arrays, outname)

def RnD_txt_to_h5(filename:str, outname:str, key=None,
                  sortby:Optional[str]='mass', unit:str='GeV',
                  feature_level:str='low_level', extra_vars:bool=True,
                  flatten:bool=False, pad_size:int=200):
    if cache and (outname is not None) and os.path.exists(outname):
        stdout.info(f'Cached output from "{outname}"')
        return 
    arrays = RnD_txt_to_arrays(filename, sortby=sortby, unit=unit,
                               feature_level=feature_level,
                               extra_vars=extra_vars,
                               flatten=True,
                               pad_size=pad_size)
    df = pd.DataFrame(arrays)
    stdout.info(f'Saving output to "{outname}"')
    if key is None:
        key = 'output'
    df.to_hdf(outname, key=key, mode='w')
    return df

def process_lhco_data(
    source: Union[str, pd.DataFrame],
    output_features: Tuple[str, ...] = ('pt', 'eta', 'phi', 'm', 'N', 'tau21', 'tau32'),
    output_parameters: Optional[Tuple[str, ...]] = ('mx', 'my'),
    signal_region: Optional[Tuple[float, float]] = (3300., 3700.),
    sortby: Optional[str] = 'mass',
    unit: str = 'TeV',
    njet: int = 2,
    label:Optional[int]=None,
):
    
    import awkward as ak
    import vector
    vector.register_awkward()

    if isinstance(source, str):
        df = pd.read_hdf(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        raise ValueError(f'Invalid input format: {type(source)}')
    jet_keys = [f'j{i}' for i in range(1, njet + 1)]
    scale = _get_unit_scale(unit)
    cols_to_scale = [f'{col}{ji}' for col in ['px', 'py', 'pz', 'pt', 'm'] for ji in jet_keys]
    cols_to_scale.append('mjj')
    cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    # convert the features to the appropriate unit
    df[cols_to_scale] = df[cols_to_scale] * scale

    j_p4 = {}
    # compute missing features
    for ji in jet_keys:
        if {f'px{ji}', f'py{ji}', f'pz{ji}'}.issubset(df.columns):
            j_p4[ji] = ak.zip({
                'px': df[f'px{ji}'],
                'py': df[f'py{ji}'],
                'pz': df[f'pz{ji}'],
                'm': df[f'm{ji}']
            }, with_name="Momentum4D")
            if f'pt{ji}' not in df.columns:
                df[f'pt{ji}'] = j_p4[ji].pt
                df[f'eta{ji}'] = j_p4[ji].eta
                df[f'phi{ji}'] = j_p4[ji].phi
                df[f'N{ji}'] = 0
        if {f'tau1{ji}', f'tau2{ji}'}.issubset(df.columns):
            tau_mask = np.array((df[f'tau1{ji}'] > 0 ) & (df[f'tau2{ji}'] > 0 ))
            df[f'tau21{ji}'] = np.where(tau_mask, np.divide(df[f'tau2{ji}'], df[f'tau1{ji}'], where=tau_mask), 0)
            df[f'tau32{ji}'] = np.where(tau_mask, np.divide(df[f'tau3{ji}'], df[f'tau2{ji}'], where=tau_mask), 0)

    # Signal Region: 3.3 TeV < mjj < 3.7 TeV
    if signal_region is not None:
        sr_min, sr_max = signal_region
        if 'mjj' not in df.columns:
            if ('j1' in j_p4) and ('j2' in j_p4):
                jj_p4 = j_p4['j1'] + j_p4['j2']
            else:
                raise RuntimeError('No information about the dijet invariant mass was found.')
            df['mjj'] = jj_p4.m
        df = df[(sr_min * scale < df['mjj']) & (df['mjj'] < sr_max * scale)]
        if df.empty:
            return df

    # filter by label for mixed dataset
    if (label is not None) and 'label' in df.columns:
        df = df[df['label'] == label]

    if sortby == 'mass':
        sort_col = 'm'
    elif sortby == 'pt':
        sort_col = 'pt'
    else:
        raise ValueError('sortby must be either "mass" or "pt"')

    def get_feature_columns(feature: str):
        return [f"{feature}{ji}" for ji in jet_keys]
        
    sort_arrays = df[get_feature_columns(sort_col)].values
    sort_indices = np.flip(np.argsort(sort_arrays, axis=-1), axis=-1)
    feature_arrays = np.zeros(shape=(sort_arrays.shape[0], len(output_features), njet))
    all_columns = []
    for i, feature in enumerate(output_features):
        feature_columns = get_feature_columns(feature)
        arrays = df[feature_columns].values
        arrays = np.take_along_axis(arrays, sort_indices, axis=-1)
        feature_arrays[:, i, :] = arrays
        all_columns.extend(feature_columns)
    feature_arrays = feature_arrays.reshape(feature_arrays.shape[0], -1)
    
    # physics parameters (for parametric signals only)
    if (output_parameters is not None) and set(output_parameters).issubset(df.columns):
        parameter_arrays = df[list(output_parameters)].values
        arrays = np.column_stack((feature_arrays, parameter_arrays))
        all_columns.extend(list(output_parameters))
    else:
        arrays = feature_arrays

    return pd.DataFrame(arrays, columns=all_columns)

def get_param_points(mass_range:Tuple[float, float]=MASS_RANGE,
                     interval:float=MASS_INTERVAL):
    unique_masses = np.arange(mass_range[0], mass_range[1] + interval, interval)
    unique_masses = unique_masses.astype('float64')
    return cartesian_product(unique_masses, unique_masses)
    
def create_high_level_dedicated_datasets(
    sample:str,
    datadir:str=DEFAULT_DATADIR,
    cache: bool = True,
    features: Tuple[str, ...] = ('pt', 'eta', 'phi', 'm', 'N', 'tau21', 'tau32'),
    parameters: Tuple[str, ...] = ('mx', 'my'), 
    signal_region: Optional[Tuple[float, float]] = (3300., 3700.),
    sortby: Optional[str] = 'mass',
    unit: str = 'TeV',
    njet: int = 2,
    num_shards:int=NUM_SHARDS,
    max_size: int = -1,
    parallel:int=-1,
    custom_slice: Optional = ...,
):
    from aliad.interface.tensorflow import TFRecordMaker
    sample = Sample.parse(sample)
    path_manager = PathManager(directories={"dataset": datadir})
    dirname = path_manager.get_directory('original_dataset')
    basename = os.path.basename(SampleURLs[sample])
    dataset_path = os.path.join(dirname, basename)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Dataset not found for the sample {sample} in {dataset_path}')
    label = sample.supervised_label
    
    df = process_lhco_data(
        dataset_path,
        output_features=features,
        output_parameters=parameters,
        sortby=sortby,
        unit=unit,
        njet=njet,
        label=label
    )

    if df.empty:
        raise RuntimeError('No events passed through preselection')

    feature_columns = [f"{feature}j{i}" for i in range(1, njet + 1) for feature in features]
    # shape = (N_event, N_jet, N_feature)
    feature_array = df[feature_columns].values.reshape(-1, njet, len(features))
    from pdb import set_trace
    set_trace()
    if set(parameters).issubset(df.columns):
        param_array = df[list(parameters)].values
        param_points = np.unique(param_array, axis=0)
    else:
        param_array = None
        param_points = get_param_points()

    for param_point in param_points[custom_slice]:
        data = {}
        if param_array is not None:
            mask = np.all(param_array == param_point, axis=1)
            selected_feature_array = feature_array[mask]
        else:
            selected_feature_array = feature_array
        if max_size > 0:
            selected_feature_array = selected_feature_array[:max_size]
        data['jet_features'] = selected_feature_array
        data['param_masses'] = np.tile(param_point, [selected_feature_array.shape[0], 1])
        data['label'] = np.tile([label], [selected_feature_array.shape[0], 1])
        # note this is hard-coded at the moment
        m1, m2 = param_point
        filename = path_manager.get_file("dedicated_dataset", feature_level="high_level",
                                         sample=sample.key, m1=int(m1), m2=int(m2),
                                         partial_format=True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        stdout.info(f"Creating dedicated dataset for the sample {sample.key} (m1={int(m1)}, m2={int(m2)})")
        tfrecord_maker = TFRecordMaker(filename, data, num_shards=num_shards,
                                       cache=cache, parallel=parallel,
                                       verbosity=stdout.verbosity)
        tfrecord_maker.run()

def create_parameterised_dataset(shard_index:int, 
                                 feature_level:str=DEFAULT_FEATURE_LEVEL,
                                 samples:Optional[List[str]]=None,
                                 datadir:str=DEFAULT_DATADIR,
                                 cache:bool=True,
                                 seed:int=BASE_SEED):    
    if samples is None:
        samples = [sample for sample in Sample if sample.default_sample]
    else:
        samples = [Sample.parse(sample) for sample in samples]
    labels = [sample.supervised_label for sample in samples]
    if (1 not in labels) or (0 not in labels):
        raise ValueError('Must include a mix of signal and background samples in a parameterised dataset')
    sample_filenames = []
    path_manager = PathManager(directories={"dataset": datadir})
    for sample in samples:
        filename_expr = path_manager.get_file("dedicated_dataset", feature_level=feature_level,
                                              shard_index=shard_index,
                                              sample=sample.key, m1="*", m2="*")
        filenames = glob.glob(filename_expr)
        if not filenames:
            raise RuntimeError(f'Cannot find datasets for the sample {sample.key} with '
                               f'shard index {shard_index} from {filename_expr}')
        sample_filenames.extend(filenames)
    def get_metadata_path(sample_filename:str):
        return os.path.splitext(sample_filename)[0] + '_metadata.json'
    def get_sample_size(metadata_filename:str):
        with open(metadata_filename, 'r') as file:
            size = json.load(file)['size']
        return size        
    metadata_filenames = list(map(get_metadata_path, sample_filenames))
    with open(metadata_filenames[0], 'r') as file:
            feature_metadata = json.load(file)['features']
    sample_size = np.sum(list(map(get_sample_size, metadata_filenames)))
    decay_modes = [sample.decay_mode for sample in samples if sample.supervised_label == 1]
    decay_mode_str = PathManager._get_decay_mode_repr(decay_modes)

    outname = path_manager.get_file("param_dataset", feature_level=feature_level,
                                    shard_index=shard_index,
                                    decay_mode=decay_mode_str)
    metadata_outname = get_metadata_path(outname)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    if cache and (os.path.exists(outname) and os.path.exists(metadata_outname)):
        stdout.info(f"Cached output from {outname}")
        return
    from aliad.interface.tensorflow.dataset import (
        get_tfrecord_array_parser, get_tfrecord_dataset, tfds_to_tfrecords
    )
    stdout.info(f"Creating parameterised dataset for the decay mode {decay_mode_str} with shard index {shard_index}")
    parse_tfrecord_fn = get_tfrecord_array_parser(feature_metadata)

    ds = get_tfrecord_dataset(sample_filenames, parse_tfrecord_fn)
    ds = ds.shuffle(buffer_size=sample_size, seed=seed + shard_index,
                    reshuffle_each_iteration=False)
    metadata = tfds_to_tfrecords(ds, outname)
    with open(metadata_outname, 'w') as file:
        json.dump(metadata, file, indent=2)
    stdout.info(f"Saved output as {outname}.")

def create_parameterised_datasets(shard_indices:Optional[List[int]]=None,
                                  feature_level:str=DEFAULT_FEATURE_LEVEL,
                                  samples:Optional[List[str]]=None,
                                  datadir:str=DEFAULT_DATADIR,
                                  cache:bool=True,
                                  seed:int=BASE_SEED,
                                  parallel:int=-1):
    from quickstats.utils.common_utils import execute_multi_tasks
    # do not use GPU
    cuda_bak = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args_list = [shard_indices, repeat(feature_level), repeat(samples), repeat(datadir), repeat(cache), repeat(seed)]
    execute_multi_tasks(create_parameterised_dataset, *args_list, parallel=parallel)
    if cuda_bak is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_bak 