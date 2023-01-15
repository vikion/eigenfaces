# Projekt Eigenfaces

todo description

## Špecifikácie otázok

todo

## Popis riešenia

### Získavanie dát

* Používame knižnice [requests](https://pypi.org/project/requests/), os, [beatifulsoup4](https://pypi.org/project/beautifulsoup4/) v Pythone

Náš program na získavanie dát dostane slovník obsahujúci skratku názvu katedry a url jej stránky a pre každú katedru zavolá funkciu `findPhotos` pomocou ktorej sa získavajú fotky.

Funkcia `findPhotos` dostane na vstupe názov katedry, url stránky katedry a cestu k zložke do ktorej má ukladať fotky. Funkcia následne načíta obsah stránky katedry a v kóde tejto stránky následne hľadá všetky tagy \<a\> nachádzajúce sa v tele tagu \<table\> pričom vynecháva tie ktorých atribút href neobsahuje reťazec "javascript", tieto tagy obsahujú meno a url podstánky jednotlivých učiteľov a nepedagogických zamestnancov. Funkcia pre každého jedného zo zamestnancov zavolá funkciu `getImages`. Po stracovaní všetkých zamestnancov katedry do konzoly vypíše informáciu o ukončení získavania fotografii.

Funkcia `getImages` dostane na vstupe meno zamestnanca, url jeho podstránky, stratku názvu katedry a cestu k zložke do ktorej má ukladať fotky. Funkcia následne načíta obsah podstránky zamestnanca a v kóde tejto stránky následne hľadá všetky tagy \<img\> s hodnotou atribútu alt "photo" ktorý sa nachádza v tele tagu \<aside\> ktorého hodnota atribútu triedy je "span3", v tomto tagu sa nachádza url fotografie daného zamestnanca. Funkcia následne načíta danú fotografiu, skontroluje či tá fotografia nie je totožná s obrázkom používaným ako placeholder pre zamestnancov ktorých fotografia sa nenachádza na serveri a v prípade, že nejde o totožnú fotografiu zloží túto fotografiu do zložky ktorej cestu dostala na vstupe s názvom tvaru "[skratka názvu katedry]_[meno zamestnanca].jpg" pričom meno zamestnanca je ešte predspracované tak, že je odstránená diakritika, medzery sú nahradene podtržníkom a sú vynechané tituly.  

### Predspracovanie

* Používame knižnicu [OpenCV](https://github.com/opencv/opencv) v Pythone
* Triedy **`Face`**, **`FaceAlign`**
* Metódy `crop_head`, `detect_features`, `rotate`, `center_crop`, `process`

Program načíta vstupné fotografie a potom upraví každú fotku nasledovne:
Na fotke detekujeme oblasť tváre pomocou metódy `detectMultiscale` z OpenCV. Každá fotka obsahovala iba jednu tvár, čiže pri detekcií nenastalo veľa problémov. Ak pri detekcii bolo nájdených viacero tvárí, tak sme vybrali tú najväčšiu. Tento postup sa osvedčil ako spoľahlivý. Tvár sa vyrezala s nejakým malým okolím pre jednoduchšie zisťovanie súradníc v neskorších krokoch. 

Nasledovne sme detekovali oči z ohraničenej oblasti tváre. Pri tejto detekcii sme narazili na viacero problémov. Niektoré z problemóv sú napríklad nájdenie false positive výsledkov, nízke rozlíšenie vstupného obrázku, neschopnosť detekovať zavreté oči, viacnásobné detekovanie toho istého oka, a tak ďalej. Tieto problémy sme riešili spustením `detectMultiscale` "po viacerých úrovniach" so zmenenými parametrami. Najprv sme určili parametre, ktoré majú za dôsledok prísnu, ale kvalitnú detekciu. Ak metóda s týmito parametrami neuspela v nájdení oboch očí, tak sme metódu spustili znovu, ale s povoľnejšími parametrami, ktoré však mohli detekovať menej presne, prípadne chybne. Parametre sme upravovali podľa pozorovania medzivýsledkov, aby boli zachytnetých čo najviac tvárí na našom datasete úspešne. V prípade viacnásobnej detekcie oka sme vyberali vždy to najvnútornejšie pre najlepšiu hranicu. Napriek všetkému úsiliu sa nemusela detekcia vždy vydariť. Preto sme vyrobili jednoduchú funkciu, ktorá nám umožnila pozíciu očí pre jednotlivé fotky manuálne naklikať. Funkciu sme použili na takých fotkách, kde sa nám výsledok detekcie nepozdával alebo neboli oči detekované vôbec.

V prípade, že sa našli viac než dve oči, sme museli vybrať správny pár.
Tento pár sme skúšali vyberať viacerými spôsobmi, ako napríklad cez tzv. `levelWeights`, čo je niečo ako confidence skóre detekcie. Tento prístup nefungoval dobre na našich fotkách, preto sme po viacerých pokusoch zvolili prístup, ktorý minimalizuje rozdiel pomeru vzdialenosti očí s veľkosťou tváre a manuálne zmeraným pomerom na vzorovej fotke. Nakoniec, po úspešnom detekovaní tváre a dvoch očí sme otočili fotku tak, aby oči boli v rovine, upravili veľkosť výslednej fotky a pozíciu očí tak, aby boli všetky rovnaké.

## Spracovanie

### PCA algoritmus

* Používame knižnice:
  *  [OpenCV](https://github.com/opencv/opencv) - metódy: `imread`, `cvtColor`
  *  [NumPy](https://numpy.org/doc/stable/index.html) - metódy: `flatter`, `reshape`, `dot`, `np.linalg.lstsq`
  *  [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html) - metódy: `fetch_olivetti_faces` 

Program načíta 2 sady fotografii. Prvou sadou sú fotografie z datasetu [Olivetti](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset) z knižnice scikit-learn ktoré využívame na natrénovanie model a druhou sadou sú nami získané fotografie zamestnancou fakulty ktoré sa snažíme klasifikovať.

#### Spracovanie Olivietti fotografii a trénovanie modelu

Po načítaní fotografii z datasetu Olivietti ich zvektorizujeme (zmeníme rozmer z $n \times n$ na $n^2 \times 1$) vypočítame priemernú tvár v tomto datasete, čiže sčítame všetky tváre a predelíme výsledok ich počtom, následne pre každú tvár vypočítame rozdiel od priemernej tváre, z týchto rozdiel od priemeru vytvoríme maticu $A$ ktorej stĺpce sú vektory rozdielov obrázkou od priemeru. Následne podľa postupu na stránke [GeeksForGeeks](https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/) vypočítame eigenfaces, čiže vytvoríme si maticu $C_{ov} = A^T A$ ktorej následne vypočítame eigenvectory $v_i$, nakoniec pomocou vzorca $u_i = A v_i$ vypočítame nami hľadané eigenfaces ktoré dáme ako stĺpce do našej matice $U$.

#### Spracovanie fotografii zamestnancou fakulty

Po načítaní fotografii tvárí zamestnancou fakulty zmeníme ich kódovanie z farebného na grayscale keďže náš algoritmus nie je určený na prácu s farebnými obrázkami, tieto sivé obrázka ďalej spracovávame tak, že ich takisto ako trénovacie obrázky zvektorizujeme a od každého odpočítame priemernú tvár trénovacích obrázkou. Poslednou časťou spracovania je výpočet koeficientov jednotlivých eigenfaces pre naše fotografie tie sú výsledkou následnej rovnice $x = U w$ kde $x$ je náš rozdiel vektorizovaného obrázka od priemeru, $U$ je matica našich eigenfaces a $w$ je vektor koeficientov. Hľadaný vektor $w$ vypočítame pomocou funkcie   `np.linalg.lstsq`. Teraz keď máme vypočítanie vektory koeficientov $w$ môžeme tieto vektory použiť na analýzu podobnosti obrázkou.

#### Podobnosť párov (pre istotu, ak chceme pouzit)

S podobnosťou tvárí otestujeme porekadlo, že manžiela sa na seba podobajú. Množinu dvojíc manželských párov sme vytvorili hľadaním ľudí s rovnakým priezviskom (s koncovkou -ová pre ženy), čo samozrejme nenajde všetkých partnerov a môže nájsť pokrvných príbuzných či menovcoc, ale na otestovanie tejto hypotézy sme pracovali s touto množinou.
Vypočítame rozdiely všetkých dvojíc tvárí, a na rozdieloch nájdeme interkvartálny rozsah (IQR). Použitím vzorca  $prvý_kvartil - (1.5 * iqr)$ nájdeme dolnú hranicu a použitím  $tretí_kvartil + (1.5 * iqr)$ nájdeme hornú hranicu dát podobností. Predpokladáme, že dvojica ľudí sa povžuje za outliera základného súboru, ak sa nachádza mimo hraníc. Z pozorovaných "partnerov" bola jedna  dvojica outlierom nad hornou hranicou, znamenajúc, že sa podobali menej ako priemerná dvojica základného súboru. Ostatní "partneri" neprekročili hranice rozsahu. Napriek malému množstvu partnerov a nedostatočnej možnosti overenia "manželstva", by sme neformálnu hypotézu o bodobnsti našich manželských párov zamietli.  

## Výsledky

todo

## Autori

Michal Dokupil  
Marián Kravec  
Viktória Ondrejová  
Pavlína Ružičková  
Andrej Zelinka

## Zdroje

* [Eyes Alignment](https://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/)
