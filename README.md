# Projekt Eigenfaces

todo description

## Špecifikácie otázok

todo

## Popis riešenia

### Načítanie dát

todo

### Predspracovanie

* Používame knižnicu [OpenCV](https://github.com/opencv/opencv) v Pythone
* Triedy **`Face`**, **`FaceAlign`**
* Metódy `crop_head`, `detect_features`, `rotate`, `center_crop`, `process`

Program načíta vstupné fotografie a potom upraví každú fotku nasledovne:
Na fotke detekujeme oblasť tváre pomocou metóde `detectMultiscale` z OpenCV. Každá fotka obsahovala iba jednu tvár, čiže pri detekcií nenastalo veľa problémov. Ak pri detekcii bolo nájdených viacero tvárí, tak sme vybrali tú najväčšiu. Tento postup sa osvedčil ako spoľahlivý. Tvár sa vyrezala s nejakým malým okolím pre jednoduchšie zisťovanie súradníc v neskorších krokoch. 

Nasledovne sme detekovali oči z ohraničenej oblasti tváre. Pri tejto detekcii sme narazili na viacero problémov. Niektoré z problemóv sú napríklad nájdenie false positive výsledkov, nízke rozlíšenie vstupného obrázku, neschopnosť detekovať zavreté oči, viacnásobné detekovanie toho istého oka, a tak ďalej. Tieto problémy sme riešili spustením `detectMultiscale` "po viacerých úrovniach" so zmenenými parametrami. Najprv sme určili parametre, ktoré majú za dôsledok prísnu, ale kvalitnú detekciu. Ak metóda s týmito parametrami neuspela v nájdení oboch očí, tak sme metódu spustili znovu, ale s povoľnejšími parametrami, ktoré však mohli detekovať menej presne, prípadne chybne. Parametre sme upravovali podľa pozorovania medzivýsledkov, aby boli zachytnetých čo najviac tvárí na našom datasete úspešne. V prípade viacnásobnej detekcie oka sme vyberali vždy to najvnútornejšie pre najlepšiu hranicu. Napriek všetkému úsiliu sa nemusela detekcia vždy vydariť. Preto sme vyrobili jednoduchú funkciu, ktorá nám umožnila pozíciu očí pre jednotlivé fotky manuálne naklikať. Funkciu sme použili na takých fotkách, kde sa nám výsledok detekcie nepozdával alebo neboli oči detekované vôbec.

V prípade, že očí sa našlo viac než dve, sme museli vybrať správny pár.
Tento pár sme skúšali vyberať viacerými spôsobmi, ako napríklad cez tzv. `levelWeights`, čož je niečo ako confidence skóre detekcie. Tento prístup nefungoval dobre na našich fotkách, preto sme po viacerých pokusoch iných prístupov nakoniec zvolili prístup, ktorý minimalizuje rozdiel pomeru vzdialenosti očí s veľkosťou tváre a manuálne zmeraným pomerom na vzorovej fotke. Nakoniec po úspešnom detekovaní tváre a dvoch očí sme otočili fotku tak, aby oči boli v rovine, upravili veľkosť výslednej fotky a pozíciu očí tak, aby boli všetky rovnaké.

### Spracovanie

todo

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
