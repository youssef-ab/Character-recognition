import numpy as np
import os
from numpy import linalg as LA
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import pytesseract
from tqdm import tqdm
#import argparse

def read_image(img_path):
  img = Image.open(img_path)
  # convert image to numpy array
  data = np.asarray(img)
  return data 

def conf_matrix(pred_values, true_values):
  pred_values_arr = np.array(list(pred_values))
  true_values_arr = np.array(list(true_values))
  cf_mat = np.zeros((len(true_values_arr),len(pred_values_arr)))
  for i in range(len(pred_values_arr)):
    for j in range(len(true_values_arr)):
      cf_mat[j,i] = int(pred_values_arr[i] == true_values_arr[j])
  return cf_mat


def score(pred_values, true_values):
  cf_mat = conf_matrix(pred_values, true_values)
  id = np.eye(cf_mat.shape[0],cf_mat.shape[1])
  return LA.norm(cf_mat-id, ord= 'fro')/LA.norm(id, ord= 'fro')

# generate character annotations using the order of contours and word annotations
def bootstrap_annotations(img, word_annotations):
  num_ann = len(word_annotations)
  symbols_annotations = []
  min_area = 25
  for i in tqdm(range(num_ann)):
    # extract geometry of the ith word to crop it
    geo_i = word_annotations[i]['geometry']
    value_i = word_annotations[i]['value']
    x_min, y_min, x_max, y_max = geo_i[0][0],geo_i[0][1],geo_i[1][0],geo_i[1][1]
    # crop the ith word
    word_i = img[y_min-4: y_max+4, x_min-2: x_max+2,:]
    # convert to grayscale image for thresholding
    word_i_g = cv2.cvtColor(word_i, cv2.COLOR_BGR2GRAY)
    ret, word_i_g = cv2.threshold(word_i_g,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # find contours (symbols)
    contours, hierarchy = cv2.findContours(word_i_g,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    idx = 0
    for cnt in sorted_ctrs :
      x,y,w,h = cv2.boundingRect(cnt)
      if w*h > min_area and idx < len(value_i) :
        dic = {}
        ch = value_i[idx]
        dic['geometry'] = [[x_min+x-2, y_min-4], [x_min-2+x+w, y_max+4 ]]
        dic['value'] = ch
        symbols_annotations.append(dic)
        idx += 1

  return symbols_annotations

#generate character annotations by recognizing the letter corresponding to each contour using pytesseract library 
def bootstrap_annotations2(img, word_annotations):
  num_ann = len(word_annotations)
  symbols_annotations = []
  scores = []
  acc = []
  min_area = 25
  num_letters = 0
  num_true = 0
  for i in tqdm(range(num_ann)):
    # extract geometry of the ith word to crop it
    geo_i = word_annotations[i]['geometry']
    value_i = word_annotations[i]['value']
    num_letters += len(value_i)
    x_min, y_min, x_max, y_max = geo_i[0][0],geo_i[0][1],geo_i[1][0],geo_i[1][1]
    # crop the ith word
    word_i = img[y_min-4: y_max+4, x_min-2: x_max+2,:]
    # convert to grayscale image for thresholding
    word_i_g = cv2.cvtColor(word_i, cv2.COLOR_BGR2GRAY)
    ret, word_i_g = cv2.threshold(word_i_g,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # find contours (symbols)
    contours, hierarchy = cv2.findContours(word_i_g,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    idx = 0
    pred_values = ''
    
    for cnt in sorted_ctrs :
      x,y,w,h = cv2.boundingRect(cnt)
      if w*h > min_area :
        dic = {}
        ch_img = word_i[:, x: x+w,:]
        ch = pytesseract.image_to_string(ch_img, lang='eng', \
        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        is_in = value_i.find(ch[0])
        num_true += int(is_in != -1)
        dic['geometry'] = [[x_min+x-2, y_min-4], [x_min-2+x+w, y_max+4 ]]
        dic['value'] = ch[0]
        pred_values = pred_values + ch[0]
        symbols_annotations.append(dic)
        idx += 1        
    scr = score(pred_values, value_i) 
    scores.append(scr)
  print('Scores mean: {:.6f} \tAccuracy: {:.6f}'.format(np.mean(scores), num_true/num_letters))
  return symbols_annotations


def gen_all_img_ann(images_dir = './samples/', json_path = './sample_labels.json', option = True):
    '''
    
    Option = True : for the second method "bootstrap_annotations" 
    Option = False : for the first method "bootstrap_annotations2" 
    
    '''
    D = {}
    json_file = open(json_path)
    anno_json = json.load(json_file)
    images_path = os.listdir(images_dir)
    if option :
        for image_name in images_path :
            img = read_image(images_dir + image_name)
            word_annotations = anno_json[image_name]
            symbols_annotations = bootstrap_annotations2(img, word_annotations)
            D[image_name] = symbols_annotations
        with open('./character_annonations2.json', 'w') as outfile:
            json.dump(D, outfile)
    else:
        for image_name in images_path :
            img = read_image(images_dir + image_name)
            word_annotations = anno_json[image_name]
            symbols_annotations = bootstrap_annotations(img, word_annotations)
            D[image_name] = symbols_annotations
        with open('./character_annonations.json', 'w') as outfile:
            json.dump(D, outfile)
        
def test(image_path, idx_ch, json_path = './character_annonations2.json'):
    
    json_file = open(json_path)
    anno_json = json.load(json_file)
    img = read_image(image_path)
    image_name = os.path.basename(image_path)
    sym_annotations = anno_json[image_name]
    assert(idx_ch<len(sym_annotations))
    geo_i = sym_annotations[idx_ch]['geometry']
    value_i = sym_annotations[idx_ch]['value']
    print(value_i)
    x_min, y_min, x_max, y_max = geo_i[0][0],geo_i[0][1],geo_i[1][0],geo_i[1][1] 
    word = img[y_min: y_max, x_min-60: x_max+60,:]
    word_gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
    ch_fig = img[y_min: y_max, x_min: x_max,:]
    ch_fig_gray = cv2.cvtColor(ch_fig, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(2,3))
    plt.imshow(ch_fig_gray, cmap='gray')
    plt.title('Extracted character')
    plt.axis('off')
    plt.show()

    
    plt.figure(figsize=(8,10))
    plt.imshow(word_gray, cmap='gray')
    plt.title('Word')
    plt.axis('off')
    plt.show()




testing = bool(int(input("Voulez-vous tester ou lancer l'algorithme (1/0): ")))
if testing :
    image_path = str(input("Entrer le chemin de l'image :"))
    idx_ch = int(input("Entrer l'indice du caractère: "))
    json_path = str(input('Enter le chemin du fichier des labels : '))
    test(image_path=image_path, idx_ch=idx_ch, json_path=json_path)
else :
    default = bool(int(input("Voulez-vous changer les paramètres par défault (0 ou 1): ")))
    if default :
        option = bool(int(input("Voulez-vous tester la méthode 1 ou 2 (0 ou 1): ")))
        images_dir = str(input('Entrer le chemin du dossier des images :') )  
        json_path = str(input('Entrer le chemin du fichier des labels :'))
        gen_all_img_ann(images_dir=images_dir,json_path=json_path,option=option)
    else :
        gen_all_img_ann()

"""if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument("image_path", help="Image path")
    parser.add_argument("json_path", help="json file path")
    parser.add_argument("ch", help="recognized character to show", type=int)
    args = parser.parse_args()
    anno_json if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument("image_path", help="Image path")
    parser.add_argument("json_path", help="json file path")
    parser.add_argument("ch", help="recognized character to show", type=int)
    args = parser.parse_args()
    anno_json = json.load(args.json_path)
    img = read_image(args.image_path)
    img_filename = os.path.basename(args.image_path)
    word_annotations = anno_json[img_filename]
    plt.figure(figsize=(6,10), dpi=80)
    plt.imshow(img)
    plt.title('Image of all words')
    plt.axis('off')
    plt.show()
    symbols_annotations = bootstrap_annotations2(img, word_annotations)
    assert(args.ch<len(symbols_annotations))
    geo_i = symbols_annotations[4]['geometry']
    value_i = symbols_annotations[4]['value']
    x_min, y_min, x_max, y_max = geo_i[0][0],geo_i[0][1],geo_i[1][0],geo_i[1][1] 
    word = img[y_min: y_max, x_min-40: x_max+40,:]
    letter = img[y_min: y_max, x_min: x_max,:]
    letter_gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(2,3))
    plt.imshow(letter_gray, cmap='gray')
    plt.title('Extracted letter')
    plt.axis('off')
    plt.show()
    y_min: y_max, x_min-40: x_max+40,:]
    letter = img[y_min: y_max, x_min: x_max,:]
    letter_gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(2,3))
    plt.imshow(letter_gray, cmap='gray')
    plt.title('Extracted letter')
    plt.axis('off')
    plt.show()
    """