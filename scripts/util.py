import json
import csv
import re
import os

def get_job_dir(project_name):
    d = f'./output/{project_name}/'

    os.mkdir(d)
    return d

def open_json(fpath):
    data = None
    with open(fpath) as f:
        data = json.load(f)
    return data


def write_json(fpath, data):
    with open(fpath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def npair(n):
    if n in (
        'AVG', 'DVC', 'PVR', 'PVT', 'RIH', 'RIR', 'DVA', 'AQR', 'AVM', 'PQR', 
        'PVM', 'DVB', 'PDA', 'PDB', 'ALA', 'AVL', 'RID', 'RIS', 
        'I3', 'I4', 'I5', 'I5', 'M1', 'M4', 'M5', 'MI'
    ): 
        return n
    if len(n) == 4 and n[-1] in 'LR' and n[:3] in (
        'ADA', 'ADE', 'ADF', 'ADL', 'AFD', 'AIA', 'AIB', 'AIM', 'AIN', 'AIY',
        'AIZ', 'ALM', 'ALN', 'ASE', 'ASG', 'ASH', 'ASI', 'ASJ', 'ASK', 'AUA', 
        'AVA', 'AVB', 'AVD', 'AVE', 'AVF', 'AVH', 'AVJ', 'AVK', 'AWA', 'AWB',
        'AWC', 'BAG', 'BDU', 'CAN', 'FLP',  'HSN'
        , 'LUA',
        'OLL', 'PDE', 'PHA', 'PHB', 'PHC', 'PLM', 'PLN', 'PVC', 'PVD', 'PVN', 
        'PVP', 'PVQ', 'PVW', 'RIA', 'RIB', 'RIC', 'RIF', 'RIG', 'RIM', 'RIP', 
        'RIV', 'RMF', 'RMG', 'RMH', 'SDQ', 'URB', 'URX'
    ):
        return n[:3]
    if len(n) == 4 and n[-1] in 'LR' and n[:3] in (
        'IL1', 'IL2', 'RME', 'RMD','GLR',
    ):
        return n[:3] + 'L/R'
    if len(n) == 4 and n[-1] in 'DV' and n[:3] in (
        'IL1', 'IL2', 'RME', 'RMD'
    ):
        return n[:3] + 'D/V'
    if len(n) == 5 and n[-2:] in ('DL', 'DR', 'VL', 'VR') and n[:3] in (
        'CEP', 'GLR', 'IL1', 'IL2', 'OLQ', 'RMD', 'SAA', 'SIA', 'SIB', 'SMB',
        'SMD', 'URA', 'URY'
    ):
        return n[:4]
    if len(n) == 8 and re.match('BWM-[DV][LR]0[0-8]', n):
        return 'BWM' + n[-2:] + n[4]
    if n in (
         'SABD', 'SABVL', 'SABVR',
    ):
        return n[:4]
    if n in (
        'CEPshDL', 'CEPshDR', 'CEPshVL', 'CEPshVR'
    ):
        return n[:6]
    if n[:2] in ('AS', 'VB', 'VA', 'VD') and n[2:] in map(str, range(12)):
        return n[:2] + 'n'
    if n in ('VA12', 'VD12', 'VD13'):
        return n[:2] + 'n'
    if re.match('^(DA[1-9])|(DB[1-7])|(DD[1-6])|(VC[1-6])$', n):
        return n[:2] + 'n'
    return n




