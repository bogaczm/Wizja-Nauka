# Michał Bogacz
# 279086
# Adom zaliczenie LAB4 - analiza online

# Domyślnie wybrany jest tryb koloru czarnego

# Program ma na celu umożliwienie łatwego wykrywania obiektów o różnych kolorach.
# W lewym górnym rogu pojawi się zielono czerwone światełko które poinformuje o tym czy obiekt się porusza.
#   W trybie manualnym, opisany poniżej, na ruch będzie reagował jedynie wybrany z góry kolor.
#   W trybie myszy kolor i nastawy są wybieranie wpełni ręcznie, przez kliknięcie.
# Zarówno przy odczycie z pliku jak i on-line działają tryby wyboru kolorów lub klikania.

# TRYB MANUALNY
# Wbudowane do systemu sa kolory:
#   - czarny - 'k'
#   - niebieski - 'b'
#   - żółty - 'y'
#   - czerwony - 'r'
# Równoczesnie program umożliwia segmentację i detekcję dowolnego koloru wybranego przy pomocy uzycia myszy.
# W celu wybrania odpowiedniego koloru należy na oryginalnym obrazie dwukrotnie nacisnąć przycisk myszy najeżdżając na 
# miejsce o wybranym kolorze.
# Aby wrócić do dowolnego koloru ustawionego domyślnie wystarczy wcisnąć jeden z klawiszy odpowiadających za ustawienie koloru.
# Równocześnie przy pomocy klawiszy + i - można płynnie zmieniać szerokość segmentacji po wybraniu koloru przez kliknięcie myszą.
#   Tryb ten nie jest dostępny dla domyślnych kolorów.
# W obu trybach 'kliknięcia mysza' oraz 'domyślnych kolorów' istnieje możliwość rozmywania oraz domykania obrazu który ostatecznie 
#   poddawany jest analizie. Wartości dobrane domyślnie to wartości optymalne dla przypadku przedstawionego w filmiku pokazowym.
#   - Zmiana domknięcia to klawisze strałka góra i strzałka dół.
#   - Zmiana współczynnika rozmycia to strzałka lewo i strzałka prawo.



# wybór czy kamera czy z pliku
import cv2
import numpy as np
import time

def odczytaj(event,x,y,flags,param):
    global klik, pozycja
    if event == cv2.EVENT_LBUTTONDBLCLK:
        klik = True
        pozycja = [x,y]

tryb_manual = False

klik = False
pozycja = [0,0]
treshold_high = np.array([255,0,0])
treshold_low = np.array([0,0,0])
treshold_border = 20

min_size = 700 

treshold_changes = 1000
kolor = (0,255,0)
kolor_examined = (0,0,0)

tryb_mysz = True

close = 10
gaussian = 1

first_loop = True

# all values are np.array representatnion of RGB palette
black_treshold_low = np.array([0,0,0])
black_treshold_high = np.array([40,40,40])

blue_treshold_low = np.array([50,40,0])
blue_treshold_high = np.array([150,80,10])

yellow_treshold_low = np.array([0,50,120])
yellow_treshold_high = np.array([80,150,180])

red_treshold_low = np.array([0,10,80])
red_treshold_high = np.array([60,50,180])

# wstępne wybranie trybu - czarny
treshold_high = black_treshold_high
treshold_low = black_treshold_low

source_type = [0, 'test.avi']

wideo = cv2.VideoCapture(source_type[0])

while(wideo):
    _, ramka = wideo.read()

    ramka_rozmyta= cv2.GaussianBlur(ramka,(3,3),gaussian)
    color_filter = cv2.inRange(ramka_rozmyta, treshold_low, treshold_high)
    
    # usuwanie obiektów pochodzących z szumu, czyli takich ponizej progu rozmiaru
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(color_filter, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1 
    size_filter = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            size_filter[output == i + 1] = 255

    es = np.ones((2*close+1,2*close+1),np.uint8)
    segmentacja = cv2.morphologyEx(size_filter,cv2.MORPH_CLOSE,es)

    # indykator ruchu
    cv2.circle(ramka, (25, 30), 15, kolor, thickness=10, lineType=8, shift=0)     
    # indykator trybu
    cv2.circle(ramka, (100, 30), 15, kolor_examined, thickness=10, lineType=8, shift=0)    

    cv2.imshow('1. oryginalny', ramka)
    cv2.imshow('2. rozmycie gaussowskie', ramka_rozmyta)
    cv2.imshow('3. tylko filtr koloru, po rozmyciu', color_filter)
    cv2.imshow('4. usuniecie zaklocen na bazie rozmiaru', size_filter)
    cv2.imshow('5. zamkniecie obrazu', segmentacja)

    # Analiza ruchu i pokazanie kontrolki ostrzegającej o ruchu
    if not first_loop:
        difference = cv2.absdiff(segmentacja,poprzednia)
        zmiany = (difference > 0).sum()
        if zmiany > treshold_changes:
            print(zmiany)
            kolor = (0,0,255) 
        else:
            kolor = (0,255,0)
    
    first_loop = False

    poprzednia = segmentacja

    # odczytanie koloru z kliknięcia myszką
    # kliknięcie ustawia filtrację i wykrywanie na kliknięty kolor
    # nadpisywane jest przez zmianę trybu na dowolny inny kolor 
    cv2.setMouseCallback('1. oryginalny', odczytaj)
    if klik :
        piksel = ramka[pozycja[1],pozycja[0]].astype(np.int)
        print("odczyt piksela (" + str(pozycja[0]) + "," + str(pozycja[1]) + ") = " + str(piksel) )
        kolor_examined = (int(piksel[0]), int(piksel[1]), int(piksel[2]))
        print('kolor to: B:{}, G:{}, R:{}'.format(type(kolor_examined[0]), kolor_examined[1], kolor_examined[2]))
        treshold_low = np.array(piksel - treshold_border)
        treshold_high = np.array(piksel + treshold_border)
        klik = False

        tryb_mysz = True
        tryb_manual = False

    # przypisanie odpowiednich trybów do klawiszy
    # ESC - wyjscie z programu
    k = cv2.waitKey(5) & 0xFF
    if k == 27 :
        break
    # k - ustawienie czarnego trybu
    elif k == ord('k'):
        treshold_high = black_treshold_high
        treshold_low = black_treshold_low
        kolor_examined = (0,0,0)
        tryb_mysz = False
        tryb_manual = True
        print('Klawiszowy tryb CZARNY')
    # r - ustawienie czerwonego trybu
    elif k == ord('r'):
        treshold_high = red_treshold_high
        treshold_low = red_treshold_low
        kolor_examined = (0,0,255)
        tryb_mysz = False
        tryb_manual = True
        print('Klawiszowy tryb CZERWONY')
    # y - ustawienie  zółtego trybu
    elif k == ord('y'):
        treshold_high = yellow_treshold_high
        treshold_low = yellow_treshold_low
        kolor_examined = (0,255,255)
        tryb_mysz = False
        tryb_manual = True
        print('Klawiszowy tryb ZOLTY')
    # b - ustawienie niebieskiego trybu
    elif k == ord('b'):
        treshold_high = blue_treshold_high
        treshold_low = blue_treshold_low
        kolor_examined = (255,0,0)
        tryb_mysz = False
        tryb_manual = True
        print('Klawiszowy tryb NIEBIESKI')
    # zmiana szerokości granicy progowania '+'/'-'
    elif k == ord('+') and not tryb_manual and treshold_border >= 0:
        treshold_border+=1
        treshold_low = np.array(piksel - treshold_border)
        treshold_high = np.array(piksel + treshold_border)
        print('Zwiekszono szerokosc granicy, border: {}'.format(treshold_border))
    elif k == ord('-') and not tryb_manual and treshold_border >= 0:
        treshold_border-=1
        treshold_low = np.array(piksel - treshold_border)
        treshold_high = np.array(piksel + treshold_border)
        print('Zmniejszono szerokosc granicy, border: {}'.format(treshold_border))
    # sterowanie domknieciem
    elif k == ord('w') and close >= 0 :
        close+=1
        print('Zwiększono poziom domkniecia: {}'.format(close))
    elif k == ord('s') and close > 0 :
        close-=1
        print('Zmniejszono poziom domkniecia: {}'.format(close))
    # sterowanie rozmyciem
    elif k == ord('d') and gaussian >= 1 :
        gaussian+=1
        print('Zwiększono poziom domkniecia: {}'.format(gaussian))
    elif k == ord('a') and gaussian > 1 :
        gaussian-=1
        print('Zmniejszono poziome domkniecia: {}'.format(gaussian))    

wideo.release()       
cv2.destroyAllWindows()