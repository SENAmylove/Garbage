import os
import yaml

def ensure_quoted(line):
    line = line.replace('"','')

    line = '"' + line + '"'

    return line


if __name__ == '__main__':
    
    with open('/Users/andromeda_sena/Documents/test.data', 'r') as reader:
        writer = open('/Users/andromeda_sena/Documents/test.yml','w')
        while line := reader.readline():
            if line.startswith(';') or len(line.strip()) == 0:
                continue

            if '\\' in line:
                line = line.replace('\\','/')
            

            if '=' in line:
                cols = line.split('=', 1)
                if cols[-1][-1] == '\n':
                    cols[-1] = cols[-1][:-1]
                cols[-1] = ensure_quoted(cols[-1].strip())
                writer.write(' : '.join(cols) + '\n')            
            else:
                if '#' in line:
                    line = line[:line.find('#')]
                line = line[:-1]
                line += ':\n'
                writer.write(line)
            
        writer.close()
            


    with open('/Users/andromeda_sena/Documents/test.yml','r') as r:
        di = yaml.safe_load(r)
        print(di)