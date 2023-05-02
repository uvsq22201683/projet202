'''init'''

from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt

def load(filename):
    toLoad= Image.open(filename)
    return np.asarray(toLoad)


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dct2(a):
    return sp.fft.dct( sp.fft.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return sp.fft.idct( sp.fft.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def save(a, path):
    Image.fromarray(a).save(path)


'''rgb/ycbcr'''

def to_Y_Cb_Cr(filename): 
    image = load(filename)
    Y = 0.299*image[:, :, 0] + 0.587*image[:, :, 1] + 0.114*image[:, :, 2]
    Cb = -0.1687*image[:, :, 0] - 0.3313*image[:, :, 1] + 0.5*image[:, :, 2] + 128
    Cr = 0.5*image[:, :, 0] - 0.4187*image[:, :, 1] - 0.0813*image[:, :, 2] +128
    return Y, Cb, Cr #pas des int

def to_RGB(Y, Cb, Cr): 
    R = np.array(np.rint(Y + 1.402*(Cr-128)))
    G = np.array(np.rint(Y - 0.34414*(Cb-128) - 0.71414*(Cr-128)))
    B = np.array(np.rint(Y + 1.772*(Cb-128)))
    image = np.dstack((R,G, B))

    return image


'''pad/unpad'''

def padding(image):
    im_shape = image.shape
    y = im_shape[0]//8
    y = 8 * (y+1)
    x = im_shape[1]//8
    x = 8 * (x+1)

    append1 = np.zeros((im_shape[0], x-im_shape[1]), dtype=int)
    append2 = np.zeros((y-im_shape[0], x), dtype=int)
    image = np.concatenate((image, append1), axis=1)
    image = np.concatenate((image, append2), axis=0)
    image = np.array(image)
    save(image.astype('uint8'), './test1.png')

    return image, im_shape

def unpad(image, im_shape):
    image = np.array(image[:im_shape[0], :im_shape[1]])
    return image


'''*2 / /2'''

def diviser_par_deux(C):
    im_shape = C.shape
    C1 = np.empty((im_shape[0], int(im_shape[1]/2)))
    for i in range(im_shape[0]):
        for j in range(int(im_shape[1]/2)):
            C1[i, j] = (C[i, j*2]+ C[i, j*2+1])/2
            #C1[i, j] = max(C[i, j*2], C[i, j*2+1])
    return C1

def multiplier_par_deux(C):
    im_shape = C.shape
    C1 = np.empty((im_shape[0], im_shape[1]*2))
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            C1[i, j*2] = C[i, j]
            C1[i, j*2+1] = C[i, j]
    return C1


'''decouper/assembler'''

def decouper(matrix):
    l_matrix = []
    im_shape = matrix.shape
    for i in range(0, im_shape[0], 8):
        for j in range(0, im_shape[1], 8):
            l_matrix.append(matrix[i:i+8, j:j+8])
    return l_matrix

def assembler(blocks, im_shape):
    print('a',im_shape)
    #for i in range(2):
    #    im_shape[i] = im_shape[i] //8
    #    if im_shape[i] %8 != 0:
    #        im_shape[i]  += 1
    #где то тут ошибка часть начала картинки переносится в конец

    # a refaire <- pourquo comme ca?
    blocks1 = np.empty((im_shape[0], 8, (im_shape[1]*8)))
    blocks11 = (blocks[(im_shape[1]*(0+1)), :, :])
    blocks12 = np.concatenate(blocks[((im_shape[1]*0)):(im_shape[1]*(0+1)-1), :, :], axis = 1)
    blocks1[0, :, :] = np.concatenate((blocks11, blocks12), axis = 1)

    for i in range(1, im_shape[0]):
        blocks1[i, :, :] = np.concatenate(blocks[((im_shape[1]*i)-1):(im_shape[1]*(i+1)-1), :, :], axis = 1)
    blocks = np.empty((im_shape[0]*8, im_shape[1]*8))
    blocks = np.concatenate(blocks1, axis = 0)

    return blocks


'''dct/idct'''

def appliquer_dct(l_matrix):
    for i in range(len(l_matrix)):
        #l_matrix[i] = dct2(l_matrix[i].astype('int')).astype('int')
        l_matrix[i] = dct2(l_matrix[i]).astype('int')
    return l_matrix

def appliquer_idct(l_matrix):
    
    for i in range(l_matrix.shape[0]):
        #l_matrix[i] = dct2(l_matrix[i]).astype('uint16')
        l_matrix[i] = idct2(l_matrix[i])
    return l_matrix

'''seuil'''

def seuil(blocks, s):
    for b in blocks:
        b[b < s] = 0
    return blocks


'''rle/unrle'''

def rle(blocks):
    blocks1 = []
    for b in blocks:
        b1 = []
        b = b.astype('str')
        repetition = 0
        for i in range(8):
            for j in range(8):

                if b[i, j] == '0':
                    repetition += 1
                else:
                    if repetition != 0:
                        b1.append(f'#{repetition}')
                        repetition = 0
                    b1.append(b[i, j])

        if repetition != 0:
            b1.append(f'#{repetition}')
        
        blocks1.append(b1)
    
    return blocks1

def unrle(matrix):
    matrix = matrix.split(' ')
    blocks = []
    for i in range(len(matrix)):
        if matrix[i][0] == '#':
            for j in range(int(matrix[i][1:])):
                blocks.append(0)
        else:
            blocks.append(int(matrix[i]))
    blocks = np.array(blocks)
    blocks = np.reshape(blocks, (8,8))
    return blocks


'''ressembler le tout'''

def compress(img_path, mode, nb_de_seuil):

    y, cb, cr = to_Y_Cb_Cr(img_path)
    img = [y, cb, cr]

    for i in range(3):

        if mode == 2 and i != 0:
            img[i] = diviser_par_deux(img[i])

        if i == 0:
            img[i], initial_shape = padding(img[i])
        else:
            img[i], _ = padding(img[i])


        #img[i] = appliquer_dct(decouper(img[i]))
        img[i] = img[i].astype('int')
        img[i] = decouper(img[i])
        img[i] = appliquer_dct(img[i])
        
        if mode != 0:
            #imposer le seuil
            img[i] = seuil(img[i], nb_de_seuil)

    return img[0], img[1], img[2], initial_shape


'''compress'''

def block_en_ligne(blocks):
    
    blocks = blocks.astype('str')   
    blocks = np.reshape(blocks, (blocks.shape[0], 64))
    return blocks.tolist()

def write_file(img_path, mode = 2, use_rle = 'RLE', nb_de_seuil = 30):
    #teste que -1 <mode<4
    print('s', nb_de_seuil)
    y, cb, cr, initial_shape = compress(img_path, mode, nb_de_seuil)
    y, cb, cr = np.around(y), np.around(cb, 1), np.around(cr, 1)
    f = open(img_path[:-4]+'_compressed.txt', 'w')

    f.write('SJPG\n')
    f.write(f'{initial_shape[0]} {initial_shape[1]}\n') 
    f.write(f'mode {mode}\n')
    f.write(f'{use_rle}\n')

    img = [y, cb, cr]
    for i in range(3):
        if use_rle == 'NORLE':
            img[i] = block_en_ligne(img[i])
        else:
            img[i] = rle(img[i])

        for j in img[i]:

            f.write(' '.join(j))
            f.write('\n')
    
    f.close()

write_file('test.png', mode = 1, use_rle = 'RLE', nb_de_seuil = 0)


'''decompresser'''

def str_to_array1(padding, im_shape1, lines, use_RLE):
    for i in range(2):
        im_shape1[i] = im_shape1[i] //8
        if im_shape1[i] %8 != 0:
            im_shape1[i]  += 1
    
    blocks = np.empty((int(im_shape1[0]*im_shape1[1]), 8, 8))
    for i in range(im_shape1[0]*im_shape1[1]):
        if use_RLE == 'RLE':
            try:
                blocks[i, :, :] = unrle(lines[i+padding].strip())
            except: print(i+padding)
        else:
            try:
                matrix = lines[i+padding].strip().split(' ')
                matrix = np.array(matrix)
                matrix = np.reshape(matrix, (8,8))
                
                blocks[i, :, :] = matrix
            except: print('e',i+padding)
    
    return blocks, i+padding+1

def decompresser(path):
    f = open(path, 'r')
    lines=f.readlines()
    f.close()

    im_shape = lines[1].strip().split(' ')
    im_shape[0] = int(im_shape[0])
    im_shape[1] = int(im_shape[1])
    
    im_shape_initial = im_shape.copy()

    mode = int(lines[2][5])
    use_rle = lines[3].strip()
    
    img = [0]*3
    print(img)

    padding = 5
    for i in range(3):
        if mode == 2:
            if i == 1:
                im_shape[1] = int(im_shape[1]/2)


        im_shape1 = im_shape.copy()
        img[i], padding = str_to_array1(padding, im_shape1, lines, use_rle)
        print(im_shape, im_shape1)
        print('p', padding)
            
        img[i] = appliquer_idct(img[i])

        
        img[i] = assembler(img[i], im_shape1)

        #padding et /2 ou ->/2 et puis padding<-
        # donc d'abord unpad et puis *2
        if mode == 2:
            if i != 0:
                print('mode')
                img[i] = multiplier_par_deux(img[i])

        img[i] = unpad(img[i], im_shape_initial)
        
    #Probleme! On ne voit pas le jaune quand on assemble l'image
    img = to_RGB(img[0], img[1], img[2])
    #change to_RGB
    #img =  np.dstack((img[0], img[1], img[2]))
    
    #Probleme! Trop d'artefacts
    save(img.astype('uint8'), 'new_test4.png')
 

decompresser('test_compressed.txt')