def str_syllables(s):
    k = ['V' if x in list('aeiouy') else 'C' for x in s]
    k = ''.join(k)
    syl_list = []
    while k:
        end = 0
        if(k.startswith('CVCC') or k.startswith('CCCV')):
            end = 4
        elif(k.startswith('CCV') or k.startswith('CVC') or k.startswith('VCC')):
            end = 3
        elif(k.startswith('VC') or k.startswith('CV')):
            end = 2
        elif(k.startswith('V')):
            end = 1
        else:
            #print "Syllables couldn't be computed: ", k, s
            return None
        syl_list.append(s[0:end])
        s = s[end:]
        k = k[end:]
    return syl_list

def phoneme_syllables(l):
    arp_vowels = ['AA','AE','AH','AO','AW','AY','EH','ER','EY','IH',
                    'IY','OW','OY','UH','UW']
    pk = ['V' if any(v in x for v in arp_vowels) else 'C' for x in l]
    pk = ''.join(pk)
    syl_list = []
    while pk:
        end = 0
        if(pk.startswith('CVCC') or pk.startswith('CCCV')):
            end = 4
        elif(pk.startswith('CCV') or pk.startswith('CVC') or pk.startswith('VCC')):
            end = 3
        elif(pk.startswith('VC') or pk.startswith('CV')):
            end = 2
        elif(pk.startswith('V')):
            end = 1
        else:
            #print "Syllables couldn't be computed: ", pk, syl_list, l
            return None
        syl_list.append(l[0:end])
        l = l[end:]
        pk = pk[end:]
    return syl_list

def str_phonem_match(s, p_list):
    """
    Input: string
    Output:
        [('per', [P, ER0]), ('fect', [F, EH1, K, T])]
    """
    syl_list = str_syllables(s)
    syl_p_list = phoneme_syllables(p_list[0])
    if len(syl_p_list) == len(syl_list):
        return [a for a, b in zip(syl_list, syl_p_list)] #[(a, b) for a, b in zip(syl_list, syl_p_list)]
    return None
