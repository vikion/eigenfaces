import os

def changeFileName(name):
    name = name.split(" ")
    s = ""
    for i in name:
        if i[-1] != ".":
            s += i.strip(",")+" "
    s = s.strip()
    s = s.replace("á", "a")
    s = s.replace("é", "e")
    s = s.replace("í", "i")
    s = s.replace("ó", "o")
    s = s.replace("ú", "u")
    s = s.replace("ý", "y")
    s = s.replace("ž", "z")
    s = s.replace("ť", "t")
    s = s.replace("č", "c")
    s = s.replace("š", "s")
    s = s.replace("ď", "d")
    s = s.replace("ň", "n")
    s = s.replace("ľ", "l")
    s = s.replace("ř", "r")
    s = s.replace("Ž", "Z")
    s = s.replace("Ť", "T")
    s = s.replace("Č", "C")
    s = s.replace("Š", "S")
    s = s.replace("Ď", "D")
    s = s.replace("Ň", "N")
    s = s.replace("Ľ", "L")
    s = s.replace("ô", "o")
    s = s.replace("ä", "a")
    return s

folders = ['AIN', 'AIN extra', 'hip', 'INF', 'KAFZM', 'MAT', 'KEF', 'KJFB', 'KTF']
for faculty in folders:
	path = faculty + "/"
	for filename in os.listdir(path):
		source = filename
		newName = changeFileName(filename)
		dest = path + faculty + "_" + newName
		source = path + filename

		os.rename(source, dest)

