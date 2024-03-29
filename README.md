# Projekt Eigenfaces

## Špecifikácie otázok

* Zhlukovanie podobných tvárí učiteľov fakulty FMFI
* Podobnosť párov 
* Zaradenie novej tváre do katedry

## Popis riešenia

### Získavanie dát

* [Skript](https://github.com/vikion/eigenfaces/blob/main/Webscrapper.ipynb)
* Používame knižnice [requests](https://pypi.org/project/requests/), os, [beatifulsoup4](https://pypi.org/project/beautifulsoup4/) v Pythone

Náš program na získavanie dát dostane slovník obsahujúci skratku názvu katedry a url jej stránky, a pre každú katedru zavolá funkciu `findPhotos` pomocou ktorej sa získavajú fotky.

Funkcia `findPhotos` dostane na vstupe názov katedry, url stránku katedry a cestu k zložke, do ktorej má ukladať fotky. Funkcia následne načíta obsah stránky katedry a v kóde tejto stránky hľadá všetky tagy \<a\> nachádzajúce sa v tele tagu \<table\> pričom vynecháva tie, ktorých atribút href neobsahuje reťazec "javascript". Tieto tagy obsahujú meno a url podstánky jednotlivých učiteľov a nepedagogických zamestnancov. Funkcia pre každého jedného zo zamestnancov zavolá funkciu `getImages`. Po stracovaní všetkých zamestnancov katedry do konzoly vypíše informáciu o ukončení získavania fotografií.

Funkcia `getImages` dostane na vstupe meno zamestnanca, url jeho podstránky, stratku názvu katedry a cestu k zložke, do ktorej má ukladať fotky. Funkcia následne načíta obsah podstránky zamestnanca a v kóde tejto stránky hľadá všetky tagy \<img\> s hodnotou atribútu alt "photo", ktorý sa nachádza v tele tagu \<aside\>, ktorého hodnota atribútu triedy je "span3", v tomto tagu sa nachádza url fotografie daného zamestnanca. Funkcia následne načíta danú fotografiu a skontroluje, či fotografia nie je totožná s obrázkom používaným ako placeholder pre zamestnancov, ktorých fotografia sa nenachádza na serveri. V prípade, že nejde o placeholder, uloží túto fotografiu do zložky, ktorej cestu dostala na vstupe s názvom tvaru "[skratka názvu katedry]_[meno zamestnanca].jpg", kde meno zamestnanca je ešte predspracované tak, že je odstránená diakritika, medzery sú nahradene podtržníkom a sú vynechané tituly.  

### Predspracovanie

* [Skript](https://github.com/vikion/eigenfaces/blob/main/FaceAlign.py)
* Používame knižnicu [OpenCV](https://github.com/opencv/opencv) v Pythone
* Triedy **`Face`**, **`FaceAlign`**
* Metódy `crop_head`, `detect_features`, `rotate`, `center_crop`, `process`

Program načíta vstupné fotografie a potom upraví každú fotku nasledovne:
Na fotke detekujeme oblasť tváre pomocou metódy `detectMultiscale` z OpenCV. Každá fotka obsahovala iba jednu tvár, čiže pri detekcií nenastalo veľa problémov. Ak pri detekcii bolo nájdených viacero tvárí, tak sme vybrali tú najväčšiu. Tento postup sa osvedčil ako spoľahlivý. Tvár sa vyrezala spolu s malým okolím pre jednoduchšie zisťovanie súradníc v neskorších krokoch. 

Nasledovne sme detekovali oči z ohraničenej oblasti tváre. Pri tejto detekcii sme narazili na viacero problémov. Niektoré z problemóv sú napríklad nájdenie false positive výsledkov, nízke rozlíšenie vstupného obrázku, neschopnosť detekovať zavreté oči, viacnásobné detekovanie toho istého oka, a tak ďalej. Tieto problémy sme riešili spustením `detectMultiscale` "po viacerých úrovniach" so zmenenými parametrami. Najprv sme určili parametre, ktoré majú za dôsledok prísnu, ale kvalitnú detekciu. Ak metóda s týmito parametrami neuspela v nájdení oboch očí, tak sme metódu spustili znovu, ale s povoľnejšími parametrami, ktoré však mohli detekovať menej presne, prípadne chybne. Parametre sme upravovali podľa pozorovania medzivýsledkov, aby bolo úspešne zachytnetých čo najviac tvárí. V prípade viacnásobnej detekcie oka sme vyberali vždy to najvnútornejšie pre najlepšiu hranicu. Napriek všetkému úsiliu sa nemusela detekcia vždy vydariť. Preto sme vyrobili jednoduchú funkciu, ktorá nám umožnila pozíciu očí pre jednotlivé fotky manuálne naklikať. Funkciu sme použili na takých fotkách, kde sa nám výsledok detekcie nepozdával alebo neboli oči detekované vôbec.

V prípade, že sa našli viac než dve oči, museli sme vybrať správny pár.
Tento pár sme skúšali vyberať viacerými spôsobmi, ako napríklad cez tzv. `levelWeights`, čo môžme chápať ako confidence skóre detekcie. Tento prístup nefungoval dobre na našich fotkách, preto sme po viacerých pokusoch zvolili prístup, ktorý minimalizuje rozdiel pomeru vzdialenosti očí s veľkosťou tváre a manuálne zmeraným pomerom na vzorovej fotke. Nakoniec, po úspešnom detekovaní tváre a dvoch očí sme otočili fotku tak, aby oči boli v rovine, upravili veľkosť výslednej fotky a pozíciu očí tak, aby boli všetky rovnaké.

### Spracovanie

#### PCA algoritmus

* Používame knižnice:
  *  [OpenCV](https://github.com/opencv/opencv) - metódy: `imread`, `cvtColor`
  *  [NumPy](https://numpy.org/doc/stable/index.html) - metódy: `flatter`, `reshape`, `dot`, `np.linalg.lstsq`
  *  [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html) - metódy: `fetch_olivetti_faces` 

Na spracovanie sme skúšali viacero postupov. Prvým je postup PCA algoritmu zo stránky [GeeksForGeeks](https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/) a druhý je inšpirovaný videom z [YouTube](https://www.youtube.com/watch?v=SsNXg6KpLSU). Nakoniec sme sa dopracovali ku 4 modelom, pričom každý z nich je trénovaný na iných dátach. Program načíta 2 sady fotografií; prvou sadou sú fotografie z datasetu [Olivetti](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset) z knižnice scikit-learn, ktoré využívame na natrénovanie modelu a druhou sadou sú nami získané fotografie zamestnancov fakulty, ktoré sa snažíme klasifikovať.

* [Model 1](https://github.com/vikion/eigenfaces/blob/main/PCA_orig_TrainSet.ipynb) - trénovaný na datasete *Olivetti* s použitím prvého postupu
* [Model 2](https://github.com/vikion/eigenfaces/blob/main/PCA_SVD.ipynb) - trénovaný na datasete *Olivetti* s použitím druhého postupu
* [Model 3](https://github.com/vikion/eigenfaces/blob/main/PCA_SVD_teachers.ipynb) - trénovaný na datasete fotiek učiteľov s použitím druhého postupu
* [Model 4](https://github.com/vikion/eigenfaces/blob/main/PCA_SVD_teachers_and_olivetti.ipynb) - trénovaný na datasete fotiek učiteľov zlúčeným s *Olivetti* s použitím druhého postupu

#### Prvý postup

##### Spracovanie Olivetti fotografii a trénovanie modelu

Po načítaní fotografií z datasetu Olivetti ich zvektorizujeme (zmeníme rozmer z $n \times n$ na $n^2 \times 1$) a vypočítame priemernú tvár v tomto datasete, čiže sčítame všetky tváre a výsledok predelíme ich počtom. Následne pre každú tvár vypočítame rozdiel od priemernej tváre, z týchto rozdielov od priemeru vytvoríme maticu $A$, ktorej stĺpce sú vektory rozdielov obrázkov od priemeru. Následne podľa postupu na stránke [GeeksForGeeks](https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/) vypočítame eigenfaces, čiže vytvoríme si maticu $C_{ov} = A^T A$, ktorej následne vypočítame eigenvectory $v_i$. Nakoniec pomocou vzorca $u_i = A v_i$ vypočítame nami hľadané eigenfaces, ktoré tvoria stĺpce našej matice $U$.

##### Spracovanie fotografii zamestnancov fakulty

Po načítaní fotografii tvárí zamestnancov fakulty zmeníme ich kódovanie z farebného na grayscale, keďže náš algoritmus nie je určený na prácu s farebnými obrázkami. Tieto sivé obrázky ďalej spracovávame tak, že ich takisto ako trénovacie obrázky zvektorizujeme a od každého odpočítame priemernú tvár trénovacích obrázkov. Poslednou časťou spracovania je výpočet koeficientov jednotlivých eigenfaces pre naše fotografie. Tie sú výsledkom následnej rovnice $x = U w$, kde $x$ je rozdiel vektorizovaného obrázka od priemeru, $U$ je matica našich eigenfaces a $w$ je vektor koeficientov. Hľadaný vektor $w$ vypočítame pomocou funkcie `np.linalg.lstsq`. Keď máme vypočítané vektory koeficientov $w$, použijeme ich na analýzu podobnosti obrázkov.

#### Druhý postup

Dáta načítavame podobne ako pri prvom postupe. Z načítaných dát vypočítame SVD (funkcia `np.linalg.svd`). Z fotiek učiteľov získame vyjadrenie ich vektorov vzhľadom na dvojrozmernú bázu vytvorenú z niektorých eigenfaces (matica $U$). Ako bázu sme skúšali viacero dvojíc, pričom sme vybrali hodnoty 6 a 7, keďže väčšina fotografií ľudí, ku ktorým sme získali viac fotiek, bola zobrazená pomerne blízko pri sebe (vzhľadom na rozptyl celého "mraku"). Na záver sme použili podobné zhlukovanie ako pri prvom postupe.

##### Podobnosť párov

* [Skript](https://github.com/vikion/eigenfaces/blob/main/marriage.py)

S podobnosťou tvárí otestujeme porekadlo, že manžiela sa na seba podobajú. Množinu dvojíc manželských párov sme vytvorili hľadaním ľudí s rovnakým priezviskom (s koncovkou -ová pre ženy), čo samozrejme nenajde všetkých partnerov a môže nájsť pokrvných príbuzných či menovcoc, ale na otestovanie tejto hypotézy sme pracovali s touto množinou.
Vypočítame rozdiely všetkých dvojíc tvárí, a na rozdieloch nájdeme interkvartálny rozsah (IQR). Použitím vzorca  $prvý kvartil - (1.5 * iqr)$ nájdeme dolnú hranicu a použitím  $tretí kvartil + (1.5 * iqr)$ nájdeme hornú hranicu dát podobností. Predpokladáme, že dvojica ľudí sa povžuje za outliera základného súboru, ak sa nachádza mimo hraníc. Z pozorovaných "partnerov" bola jedna  dvojica outlierom nad hornou hranicou, znamenajúc, že sa podobali menej ako priemerná dvojica základného súboru. Ostatní "partneri" neprekročili hranice rozsahu. Napriek malému množstvu partnerov a nedostatočnej možnosti overenia "manželstva", by sme neformálnu hypotézu o --podobnsti našich manželských párov zamietli.

##### Zaradenie novej tváre do katedry

* [Skript](https://github.com/vikion/eigenfaces/blob/main/average_face_katedra.py)

Nakoniec sme vytvorili program, ktorý podľa fotky tváre nájde takú katedru, kde sa priemer tvárí jej členov najviac zhoduje s fotkou na vstupe. 
Najprv bolo potrebné vytvoriť priemerné tváre jednotlivých katedier. Fotografie sme si vďaka skratky katedry v názve súboru vedeli rozdeliť do polí a následne sme si pomocou knižnice OpenCV a jej funkcie `addWeighted()` vytvorili jednotlivé priemery.
Výsledný program už porovnáva fotografiu na vstupe so všetkými priemernými tvárami katedier. Výstupom je názov katedry, s ktorou má vstupná fotka najmenšiu Strednú kvadratickú chybu (Mean squared error).

## Výsledky

"Lakeť", podľa ktorého sme určili počet zhlukov:  
![alt text](./clusters/elbow.png)

Nájdené zhluky:  
![alt text](./clusters/scatter.png)

Tu máme zobrazenú podmnožinu nájdených zhlukov:

#### Zhluk 1
![alt text](./clusters/output1.png)
#### Zhluk 2
![alt text](./clusters/output2.png)
#### Zhluk 3
![alt text](./clusters/output3.png)
#### Zhluk 4
![alt text](./clusters/output4.png)


## Autori

Michal Dokupil  
Marián Kravec  
Viktória Ondrejová  
Pavlína Ružičková  
Andrej Zelinka

## Zdroje

* [Eyes Alignment](https://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/)
* [Prvý postup (GeeksForGeeks)](https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/)
* Videoprednášky z University of Washington: [Eigenfaces 2](https://youtu.be/yYdYrAKghF4) a [Eigenfaces 3](https://youtu.be/SsNXg6KpLSU)
* [Olivetti](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset)
* Obrázky učiteľov [FMFI](https://fmph.uniba.sk/pracoviska/) a obrázky učiteľov [KTVŠ](http://ktvs.fmph.uniba.sk/article_forms.php?section=9&article_id=195)
* [Priemer tvárí](https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/)
