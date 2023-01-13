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
