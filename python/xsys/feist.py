import sys

def feiphi(kf, sf, df, m):
    key, _key = getKFF(kf)
    src = getSFF(sf, m)
    kg = KG(key)
    c = PSUC(kg)
    results = []
    for s in src:
        if m == "-e":
            results.append(c.en(s))
        else:
            results.append(c.de(s))
    return _key, results

def srtf(key, results):
    counter = 0
    char_tup = ""
    out = [key,'\n']
    for r in results:
        for c in r:
            char_tup += c
            counter += 1
            if (counter == 2):
                value = int(char_tup, 16)
                if value != 0:
                    char_tup = chr(value)
                    out.append(char_tup)
                counter = 0
                char_tup = ""
    return out

def saveResultsToFile(dest_file, results, mode):
    try:
        file = open(dest_file, 'w')
    except IOError:
        print("Error: Can't create the \"" + dest_file + "\" file.")
    with file:
        counter = 0
        char_tup = ""
        for r in results:
            if mode == "-e":
                file.write(r)
            else:
                for c in r:
                    char_tup += c
                    counter += 1
                    if (counter == 2):
                        value = int(char_tup, 16)
                        if value != 0:
                            char_tup = chr(value)
                            file.write(char_tup)
                        counter = 0
                        char_tup = ""

class KG:
    constant_64bit = pow(2, 63)

    def __init__(self, key):
        self.original_key = int(key, 16)
        self.keys = self.generateKeys(self.original_key)

    def getKeys(self):
        return self.keys

    def getOriginalKey(self):
        return self.original_key

    def shiftBitLeft(self, block):
        if block >= self.constant_64bit:
            block -= self.constant_64bit
            block = block << 1
            block += 1
        else:
            block = block << 1
        return block

    def createKeyArray(self, block):
        key_array = []
        for i in range(0, 64):
            block = self.shiftBitLeft(block)
            key_array.append(block)

        return key_array

    def breakHexIntoChunks(self, block):
        block = hex(block)[2:]
        if (len(block) > 16):
            block = block[:16]
        else:
            while (len(block) < 16):
                block = "0" + block

        block_list = []
        i = len(block)
        while (i > 0):
            byte = block[i - 2: i]
            block_list.append(byte)
            i -= 2
        return block_list

    def generateKeys(self, key):
        row_keys = self.createKeyArray(key)
        keys = []
        counter = -1
        x = 0
        while (x < 192 and counter < 16):
            i = x % 64
            byte_arr = self.breakHexIntoChunks(row_keys[i])
            if (x % 12 == 0):
                keys.append([])
                counter += 1

            b = x % 4
            if counter % 2 == 1:
                b += 4
            keys[counter].append(int(byte_arr[b], 16))
            x += 1

        return keys

    def printKeys(self, keys):
        # add asertions that keys have size of 16x12
        for key in keys:
            for b in key:
                print(hex(b), end=' ')
            print()

class PSUC:
    const_2_16 = 2 ** 16
    ftable = [0xa3, 0xd7, 0x09, 0x83, 0xf8, 0x48, 0xf6, 0xf4, 0xb3, 0x21, 0x15, 0x78, 0x99, 0xb1, 0xaf, 0xf9,
              0xe7, 0x2d, 0x4d, 0x8a, 0xce, 0x4c, 0xca, 0x2e, 0x52, 0x95, 0xd9, 0x1e, 0x4e, 0x38, 0x44, 0x28,
              0x0a, 0xdf, 0x02, 0xa0, 0x17, 0xf1, 0x60, 0x68, 0x12, 0xb7, 0x7a, 0xc3, 0xe9, 0xfa, 0x3d, 0x53,
              0x96, 0x84, 0x6b, 0xba, 0xf2, 0x63, 0x9a, 0x19, 0x7c, 0xae, 0xe5, 0xf5, 0xf7, 0x16, 0x6a, 0xa2,
              0x39, 0xb6, 0x7b, 0x0f, 0xc1, 0x93, 0x81, 0x1b, 0xee, 0xb4, 0x1a, 0xea, 0xd0, 0x91, 0x2f, 0xb8,
              0x55, 0xb9, 0xda, 0x85, 0x3f, 0x41, 0xbf, 0xe0, 0x5a, 0x58, 0x80, 0x5f, 0x66, 0x0b, 0xd8, 0x90,
              0x35, 0xd5, 0xc0, 0xa7, 0x33, 0x06, 0x65, 0x69, 0x45, 0x00, 0x94, 0x56, 0x6d, 0x98, 0x9b, 0x76,
              0x97, 0xfc, 0xb2, 0xc2, 0xb0, 0xfe, 0xdb, 0x20, 0xe1, 0xeb, 0xd6, 0xe4, 0xdd, 0x47, 0x4a, 0x1d,
              0x42, 0xed, 0x9e, 0x6e, 0x49, 0x3c, 0xcd, 0x43, 0x27, 0xd2, 0x07, 0xd4, 0xde, 0xc7, 0x67, 0x18,
              0x89, 0xcb, 0x30, 0x1f, 0x8d, 0xc6, 0x8f, 0xaa, 0xc8, 0x74, 0xdc, 0xc9, 0x5d, 0x5c, 0x31, 0xa4,
              0x70, 0x88, 0x61, 0x2c, 0x9f, 0x0d, 0x2b, 0x87, 0x50, 0x82, 0x54, 0x64, 0x26, 0x7d, 0x03, 0x40,
              0x34, 0x4b, 0x1c, 0x73, 0xd1, 0xc4, 0xfd, 0x3b, 0xcc, 0xfb, 0x7f, 0xab, 0xe6, 0x3e, 0x5b, 0xa5,
              0xad, 0x04, 0x23, 0x9c, 0x14, 0x51, 0x22, 0xf0, 0x29, 0x79, 0x71, 0x7e, 0xff, 0x8c, 0x0e, 0xe2,
              0x0c, 0xef, 0xbc, 0x72, 0x75, 0x6f, 0x37, 0xa1, 0xec, 0xd3, 0x8e, 0x62, 0x8b, 0x86, 0x10, 0xe8,
              0x08, 0x77, 0x11, 0xbe, 0x92, 0x4f, 0x24, 0xc5, 0x32, 0x36, 0x9d, 0xcf, 0xf3, 0xa6, 0xbb, 0xac,
              0x5e, 0x6c, 0xa9, 0x13, 0x57, 0x25, 0xb5, 0xe3, 0xbd, 0xa8, 0x3a, 0x01, 0x05, 0x59, 0x2a, 0x46]

    def __init__(self, key_generator):
        self.original_key = key_generator.getOriginalKey()
        self.keys = key_generator.getKeys()

    def whitening(self, word, key):
        r = []

        key = format(key, "016x")
        for i in range(0, 4):
            j = i * 4

            sw = word[j: j + 4]
            sk = key[4 * i:4 * i + 4]
            w = int(sw, 16) ^ int(sk, 16)
            r.append(w)
        return r

    def g_function(self, r, k0, k1, k2, k3):
        r = format(r, '04x')
        g1 = int(r[:2], 16)
        g2 = int(r[2:4], 16)
        g3 = int(self.ftable[g2 ^ k0]) ^ g1
        g4 = int(self.ftable[g3 ^ k1]) ^ g2
        g5 = int(self.ftable[g4 ^ k2]) ^ g3
        g6 = int(self.ftable[g5 ^ k3]) ^ g4
        # print("g1: " + hex(g1) + ' g2: ' + hex(g2) + ' g3: ' + hex(g3) + ' g4: ' + hex(g4) + ' g5: ' + hex(g5) + ' g6: ' + hex(g6))
        return int(format(g5, '02x') + format(g6, '02x'), 16)

    def f_function(self, r0, r1, round):
        k = self.keys[round]

        # print("Keys: ", end = ' ')
        # for ki in k:
        #     print(hex(ki), end = ' ')
        # print()

        t0 = self.g_function(r0, k[0], k[1], k[2], k[3])
        t1 = self.g_function(r1, k[4], k[5], k[6], k[7])
        # print("t0: " + hex(t0) + " t1: " + hex(t1))

        k89 = int(format(k[8], '02x') + format(k[9], '02x'), 16)
        kab = int(format(k[10], '02x') + format(k[11], '02x'), 16)
        f0 = (t0 + 2 * t1 + k89) % self.const_2_16
        f1 = (2 * t0 + t1 + kab) % self.const_2_16
        return [f0, f1]

    def round_function(self, r, round):
        f = self.f_function(r[0], r[1], round)
        # print("f0: " + hex(f[0]) + " f1: " + hex(f[1]))

        new_r0 = r[2] ^ f[0]
        new_r1 = r[3] ^ f[1]
        new_r2 = r[0]
        new_r3 = r[1]
        return [new_r0, new_r1, new_r2, new_r3]

    def en(self, pt_hex):
        r = self.whitening(pt_hex, self.original_key)

        # print("\n\n\nEncryption:")
        for i in range(0, 16):
            # print("Beginning of Round: " + str(i))
            r = self.round_function(r, i)
            # print("Block: " + hex(r[0]) + hex(r[1])[2:] + hex(r[2])[2:] + hex(r[3])[2:])
            # print("End of Round: " + str(i))
            # print()

        y = [r[2], r[3], r[0], r[1]]
        y_hex = ""
        for yi in y:
            y_hex += format(yi, "04x")

        # print("y_hex = " + y_hex)
        c = self.whitening(y_hex, self.original_key)
        c_str = ""
        # print("\n\nCiphertext: 0x", end = '')
        for ci in c:
            c_str += format(ci, "04x")
        #     print(format(ci, "04x"), end = '')
        # print(c_str)
        return c_str

    def de(self, ct_hex):
        r = self.whitening(ct_hex, self.original_key)
        for i in range(15, -1, -1):
            # print("Beginning of Round: " + str(i))
            r = self.round_function(r, i)
            # print("Block: " + hex(r[0]) + hex(r[1])[2:] + hex(r[2])[2:] + hex(r[3])[2:])
            # print("End of Round: " + str(i))
            # print()

        p = [r[2], r[3], r[0], r[1]]
        p_hex = ""
        for pi in p:
            p_hex += format(pi, "04x")

        p = self.whitening(p_hex, self.original_key)
        # print("\n\nPlaintext: 0x", end = '')
        p_str = ""
        for pi in p:
            p_str += format(pi, "04x")
        # print(p_str)
        return p_str

def stringToHex(str):
    s_hex = []
    for c in str:
        s_hex.append(ord(c))

    h_str = ""
    for h in s_hex:
        h_str += format(h, "02x")
    return h_str


def getKFF(key_file):
    try:
        file = open(key_file, 'r')
    except IOError:
        print("Error: Can't open the \"" + key_file + "\" file.")
    with file:
        key = file.readline()
        _key = key
        try:
            key = int(key, 16)
            key = format(key, "016x")
            if (len(key) > 16):
                key = key[:16]
        except ValueError:
            print("Error: Can't parse the key")

        return key, _key


def getSFF(src_file, mode):
    try:
        file = open(src_file, 'r')
    except IOError:
        print("Error: Can't open the \"" + src_file + "\" file.")
    with file:
        hex_strings = []
        counter = 0
        i = -1
        while True:
            c = file.read(1)
            if not c:
                break
            if counter % 16 == 0:
                hex_strings.append("")
                i += 1
            if mode == "-e":
                c = format(ord(c), "02x")
                counter += 1
            counter += 1
            hex_strings[i] += c

        if i < 0:
            raise ValueError("Error: the source file is empty!")
        hex_strings[i] = format(int(hex_strings[i], 16), "016x")
        return hex_strings

'''
python3 feist.py <key_file> <source_file> <destination_file> <option>
<key_file>
<source_file>
<destination_file>
<option> "-e" or "-d"
'''
def main():
    argv = sys.argv[1:]
    if len(argv) < 4:
        print("Missing command line arguments!")
        return
    elif len(argv) > 4:
        print("Unknown command line arguments!")
        return
    _1,_2=feiphi(argv[0], argv[1], argv[2], argv[3])
    print("".join(srtf(_1,_2)))
    if 'e' in argv[3]:
        saveResultsToFile(argv[2], _2, argv[3])

def _(a,b,c,d,f=False):
    _1,_2=feiphi(a, b, c, d)
    if f:
        print(f'{"".join(srtf(_1,_2))}')
    if 'e' in d:
        saveResultsToFile(c, _2, d)

def test(a,b):
    _1, _2 = feiphi(a, b, '', '-d')
    _ = ''.join(srtf(_1, _2)).split('\n')
    print(f"{_}")

if __name__ == '__main__':
    if False:
        main()
    elif False:
        _('sandbox/tf0.xs', 'sandbox/tf3.xs', 'sandbox/tf4.xs', "-e")
    elif True:
        _('sandbox/tf0.xs', 'sandbox/tf4.xs', '', "-d")