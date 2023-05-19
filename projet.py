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

'''quantification'''

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]])


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
        b[abs(b) < s] = 0
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


'''modes facultatives'''

def mode3(blocks):
    for i in range(len(blocks)):
        limit =  np.sort(np.absolute(blocks[i].copy()), axis=None)[-8]
        #blocks[i] = np.where(blocks[i] < limit or blocks[i] > limit*-1, 0, blocks[i])
        blocks[i][abs(blocks[i]) < limit] = 0
    return blocks

def mode4_compress(blocks):
    global Q
    for i in range(len(blocks)):
        blocks[i] = (blocks[i]/Q).astype('int')
        #blocks[i] = blocks[i]
    return blocks

def mode4_decompress(blocks):
    global Q
    for i in range(len(blocks)):
        #blocks[i] = (blocks[i]*Q).astype('int')
        blocks[i] = (blocks[i]*Q)
    return blocks     

def Zig_zag_rle(blocks):
    print('b', len(blocks))
    blocks1 = []
    for i in range(len(blocks)):
        b1 = []
        for j in range(0, 64):
            if j < 8:
                slice = [w[:j] for w in blocks[i][:j]]
            else:
                slice = [w[j-9:] for w in blocks[i][j-9:]]
            diag = [slice[w][len(slice)-w-1] for w in range(len(slice))]
            if len(diag) % 2:
                diag.reverse()
            b1 += diag

        b2 = []
        count_0 = 0
        for j in range(0, 65):
                if b1[j] == 0:
                    count_0 += 1
                else:
                    if count_0 != 0:
                        b2.append(f'#{count_0}')
                        count_0 = 0
                    b2.append(str(b1[j]))
        if count_0 != 0:
            b2.append(f'#{count_0}')
        blocks1.append(b2)
    print('b1', len(blocks1))
    #print('b1', b1)
    return blocks1


def Zig_zag_unrle(matrix):
    
    matrix = matrix.split(' ')
    blocks = []
    for i in range(len(matrix)):
        if matrix[i][0] == '#':
            for j in range(int(matrix[i][1:])):
                blocks.append(0)
        else:
            blocks.append(int(matrix[i]))
    
    blocks1_1 = np.array([[0]*8 for _ in range(8)])
    blocks1_2 = np.array([[0]*8 for _ in range(8)])
    #blocks1[0][0] = blocks[total]
    #total+=1
    to = 8
    for b in [blocks1_1, blocks1_2]:
        total = 0
        for i in range(0,to):
            for j in range(i+1):
                if i%2==0:
                    b[i-j, j] = blocks[total]
                    total += 1
                else:
                    b[j, i-j] = blocks[total]
                    total += 1
        to -= 1
        blocks = blocks[total:]
        blocks.reverse()
    
    result = np.maximum(blocks1_1, blocks1_2[::-1,::-1])
      
    return result

'''ressembler le tout'''

def compress(img_path, mode, nb_de_seuil):
    
    y, cb, cr = to_Y_Cb_Cr(img_path)
    #save(idct2(dct2(cr[8*8:8*9, 8*9:8*10])).astype('uint8'), 'a.png')
    img = [y, cb, cr]
    save(idct2(dct2(y)).astype('uint8'), 'a.png')

    for i in range(3):

        if mode >= 2 and i != 0:
            img[i] = diviser_par_deux(img[i])

        if i == 0:
            img[i], initial_shape = padding(img[i])
        else:
            img[i], _ = padding(img[i])


        #img[i] = appliquer_dct(decouper(img[i]))
        #img[i] = img[i].astype('int')
        img[i] = decouper(img[i])
        img[i] = appliquer_dct(img[i])
        
        #if mode != 0:
        if mode == 1 or mode == 2:
            #imposer le seuil
            img[i] = seuil(img[i], nb_de_seuil)
            print('seuil', nb_de_seuil)
            
        elif mode == 3:
            img[i] = mode3(img[i])
        elif mode == 4:
            img[i] = mode4_compress(img[i])
        

    return img[0], img[1], img[2], initial_shape


'''compress'''

def block_en_ligne(blocks):
    
    blocks = np.array(blocks)
    blocks = blocks.astype('str')   
    blocks = np.reshape(blocks, (blocks.shape[0], 64))
    return blocks.tolist()

def write_file(img_path, mode = 2, use_rle = 'RLE', nb_de_seuil = 30):
    #teste que -1 <mode<4
    print('s', nb_de_seuil)
    y, cb, cr, initial_shape = compress(img_path, mode, nb_de_seuil)
    #y, cb, cr = np.around(y), np.around(cb, 1), np.around(cr, 1)
    f = open(img_path[:-4]+'_compressed1.txt', 'w')

    f.write('SJPG\n')
    f.write(f'{initial_shape[0]} {initial_shape[1]}\n') 
    f.write(f'mode {mode}\n')
    f.write(f'{use_rle}\n')

    img = [y, cb, cr]
    for i in range(3):
        if use_rle == 'NORLE':
            img[i] = block_en_ligne(img[i])
        elif use_rle == 'RLE':
            img[i] = rle(img[i])
        elif use_rle == 'ZIGZAG_RLE':
            img[i] = Zig_zag_rle(img[i])

        for j in img[i]:
            f.write(' '.join(j))
            f.write('\n')
    
    f.close()

write_file('test.png', mode = 4, use_rle = 'RLE', nb_de_seuil = 4)


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
        elif use_RLE == 'ZIGZAG_RLE':
            try:
                blocks[i, :, :] = Zig_zag_unrle(lines[i+padding].strip())
            except: 
                print('aaaaa',i+padding)
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
    #print(img)

    padding = 5
    #for i in range(3):
    for i in range(3):
        if mode >= 2:
            if i == 1:
                im_shape[1] = int(im_shape[1]/2)


        im_shape1 = im_shape.copy()
        img[i], padding = str_to_array1(padding, im_shape1, lines, use_rle)
        print(im_shape, im_shape1)
        print('p', padding)

        if i == 0:
            print(img[i][100])
        
        if mode == 4:
            img[i] = mode4_decompress(img[i])
        #print(img[i])

        img[i] = appliquer_idct(img[i])

        
        img[i] = assembler(img[i], im_shape1)
        

        #padding et /2 ou ->/2 et puis padding<-
        # donc d'abord unpad et puis *2
        if mode >= 2:
            if i != 0:
                print('mode')
                img[i] = multiplier_par_deux(img[i])

        img[i] = unpad(img[i], im_shape_initial)
        
    #Probleme! On ne voit pas le jaune quand on assemble l'image
    img = to_RGB(img[0], img[1], img[2])
    #change to_RGB
    #img =  np.dstack((img[0], img[1], img[2]))
    
    #Probleme! Trop d'artefacts
    save(img.astype('uint8'), 'new_test5.png')
 

decompresser('test_compressed1.txt')

print(psnr(load('test.png'), load('new_test5.png')))


A = [[1,2,3,4, 51, 52, 53, 54],
     [5,6,7,8, 51, 52, 53, 54],
     [9,10,11,12, 51, 52, 53, 54],
     [13,14,15,16, 51, 52, 53, 54],
     [13,14,15,16, 51, 52, 53, 54],
     [13,14,15,16, 51, 52, 53, 54],
     [13,14,15,16, 51, 52, 53, 54],
     [13,14,15,16, 51, 52, 53, 54],]
#print(Zig_zag_rle([np.array(A)]))
#print(' '.join(Zig_zag_rle([np.array(A)])[0]))
#print(Zig_zag_unrle(' '.join(Zig_zag_rle([np.array(A)])[0])))


m = [[1,2,6, 7], 
     [3,5,8,11], 
     [4, 9, 10, 12]]

def zigzag(matrix):

    #Vecteurs direction du mouvement
    droite = [0,1] #+1 colonne
    bas = [1,0] #+1 ligne (vers le bas)
    bas_gauche = [1, -1] #+1 ligne -1 colonne
    haut_droite = [-1, 1] # +1 ligne +1 colonne
    direction = haut_droite 

    ligne = len(matrix)
    colonne = len(matrix[0])
    liste_zigzag = []
    i, j = 0,0

    while i < ligne and j < colonne:
        liste_zigzag.append(matrix[i][j])

        if direction == droite:
            if i == 0: 
                direction = bas_gauche
            elif i == ligne-1:
                direction = haut_droite
            elif j == colonne - 1:
                direction = bas

        elif direction == bas:
            if j==0:
                direction= haut_droite
            elif j == colonne-1:
                direction = bas_gauche
        
        elif direction == haut_droite:
            if i ==0:
                direction = droite
            elif j==colonne-1:
                direction = bas
        
        elif direction == bas_gauche:
            if i==ligne-1:
                direction = droite
            elif j == 0:
                direction = bas

        #On ajoute les coordonées des vecteurs mouvement à la position de matrix[i][j]
        i+=direction[0]
        j+=direction[1]

    return liste_zigzag

zigzag(m)

