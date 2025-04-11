import numpy as np
import os
import json
import pandas as pd

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

def compute_avg_error(psnr, ssim, lpips):
  """The 'average' error used in the paper."""
  mse = psnr_to_mse(psnr)
  dssim = np.sqrt(1 - ssim)
  return np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))

dataset = 'blender'
exp_name = '8_views'
log_path = f'./output/{dataset}/{exp_name}'
repeat = 1

gather = {}
psnr_total = []
ssim_total = []
lpips_total = []
ssimsk_total = []
avgerror_total = []

# scenes = ['fern', 'flower', 'fortress', 'horns', 'room', 'trex', 'orchids', 'leaves']
# scenes = ['scan30', 'scan34', 'scan41', 'scan45',  'scan82', 'scan103',  'scan38', 'scan21', 'scan40',  'scan55',  'scan63', 'scan31', 'scan8',  'scan110',  'scan114']
scenes = ['chair', 'drums', 'hotdog', 'ficus', 'lego', 'materials', 'mic', 'ship' ]
for scene in scenes:
  gather[f'{scene}'] = {}
  psnr_scene = []
  ssim_scene = []
  ssimsk_scene = []
  lpips_scene = []
  avg_scene = []    
  for i in range(repeat):
    json_path = os.path.join(log_path, f'run{i+1}', scene, 'results_eval.json')
    # json_path = os.path.join(log_path, f'run{i+1}', scene, 'results_eval_mask.json')
    
    with open(json_path, 'r') as f:
      data = json.load(f)
      if scene in ['chair', 'mic']:
        psnr_scene.append(data['ours_30000']['PSNR'])
        ssim_scene.append(data['ours_30000']['SSIM'])
        ssimsk_scene.append(data['ours_30000']['SSIM_sk'])
        lpips_scene.append(data['ours_30000']['LPIPS'])
      else:
        psnr_scene.append(data['ours_6000']['PSNR'])
        ssim_scene.append(data['ours_6000']['SSIM'])
        ssimsk_scene.append(data['ours_6000']['SSIM_sk'])
        lpips_scene.append(data['ours_6000']['LPIPS'])              
      avg_scene.append(compute_avg_error(psnr_scene[-1], ssimsk_scene[-1], lpips_scene[-1]))
  mean_psnr = np.mean(np.array(psnr_scene))
  mean_ssim = np.mean(np.array(ssim_scene))
  mean_lpips = np.mean(np.array(lpips_scene))
  mean_ssimsk = np.mean(np.array(ssimsk_scene))
  mean_avgerror = np.mean(np.array(avg_scene))
  gather[f'{scene}']['PSNR'] = str(round(mean_psnr, 3)) + u"\u00B1" + str(round(np.std(np.array(psnr_scene)),3))
  gather[f'{scene}']['SSIM'] = str(round(mean_ssim, 3)) + u"\u00B1" + str(round(np.std(np.array(ssim_scene)),3))
  gather[f'{scene}']['SSIM_sk'] = str(round(mean_ssimsk,3)) + u"\u00B1" + str(round(np.std(np.array(ssimsk_scene)),3))
  gather[f'{scene}']['LPIPS'] =  str(round(mean_lpips,3)) + u"\u00B1" + str(round(np.std(np.array(lpips_scene)),3))
  gather[f'{scene}']['AVG_ERROR'] = str(round(mean_avgerror,3)) + u"\u00B1" + str(round(np.std(np.array(avg_scene)),3))
  psnr_total.append(mean_psnr)
  ssim_total.append(mean_ssim)
  ssimsk_total.append(mean_ssimsk)
  lpips_total.append(mean_lpips)
  avgerror_total.append(mean_avgerror)

gather['total'] = {}
gather['total']['PSNR'] = str(round(np.mean(np.array(psnr_total)),3)) + u"\u00B1" + str(round(np.std(np.array(psnr_total)),3))
gather['total']['SSIM'] = str(round(np.mean(np.array(ssim_total)),3)) + u"\u00B1" + str(round(np.std(np.array(ssim_total)),3))
gather['total']['SSIM_sk'] = str(round(np.mean(np.array(ssimsk_total)),3)) + u"\u00B1" + str(round(np.std(np.array(ssimsk_total)),3))
gather['total']['LPIPS'] = str(round(np.mean(np.array(lpips_total)),3)) + u"\u00B1" + str(round(np.std(np.array(lpips_total)),3))
gather['total']['AVG_ERROR'] = str(round(np.mean(np.array(avgerror_total)),3)) + u"\u00B1" + str(round(np.std(np.array(avgerror_total)),3))
df = pd.DataFrame(gather)
print(df.T)
df.to_excel(f'output_{dataset}_{exp_name}.xlsx')


    