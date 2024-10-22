##coding:utf-8
import re
from grobid.grobid_parser.parse import _string_to_tree
ns = "http://www.tei-c.org/ns/1.0"

def revise_coodinates_str(xml_text, mmd_text):
    # xml_text = open(xmlfilename, "r", encoding='utf-8').read()
    tree = _string_to_tree(xml_text)
    tei = tree.getroot()
    pagesize_dict = dict()
    for (i, bs) in enumerate(tei.findall(f".//{{{ns}}}facsimile/{{{ns}}}surface")):
        # print(bs.attrib.get('n'), bs.attrib.get('lrx'), bs.attrib.get('lry'))  ###页面序号，宽度，页面高度
        pageid = bs.attrib.get('n')
        pagewidth = float(bs.attrib.get('lrx'))
        pageheight = float(bs.attrib.get('lry'))
        pagesize_dict[pageid] = [pagewidth, pageheight]

    # print(pagesize_dict)

    # mmd_text = open(mmdfilename, "r", encoding='utf-8').read()
    citation_pattern = re.compile(r'coords\=\"(.*?)\"', re.S)  ####找到所有坐标
    citation_results = citation_pattern.findall(mmd_text)
    newmmd_text = mmd_text
    for idx, cit in enumerate(citation_results):

        if len(cit) < 5:  ###不是坐标？
            # print('原坐标为空', cit)
            continue
        if float(re.split(',', cit)[-2]) < 1.0:  ####高度低于阈值，可能已经过处理
            # print('bbox过小，跳过', cit)
            continue
        newcood = ''
        coods = re.split(';', cit)
        for cood in coods:
            kys = re.split(',', cood)
            pagesize = pagesize_dict.get(kys[0])
            newlx = round(float(kys[1]) / pagesize[0], 5)
            newly = round(float(kys[2]) / pagesize[1], 5)
            # newwd = round(float(kys[3]) / pagesize[0], 5)##宽
            # newht = round(float(kys[4]) / pagesize[1], 5)##高
            newrx = round(float(kys[1]) / pagesize[0] + float(kys[3]) / pagesize[0], 5)  ##宽
            newry = round(float(kys[2]) / pagesize[1] + float(kys[4]) / pagesize[1], 5)  ##高
            if len(kys) >= 6:
                newcood += (kys[0] + ',' + str(newlx) + ',' + str(newly) + ',' + str(newrx) + ',' + str(newry) + ',' + str(kys[5])+';')
            else:
                newcood += (kys[0] + ',' + str(newlx) + ',' + str(newly) + ',' + str(newrx) + ',' + str(newry) + ';')

        # print('replace ',cit,' with ', newcood[:-1])
        newmmd_text = newmmd_text.replace(cit, newcood[:-1])  ####全文范围更换

    # newfilename = mmdfilename.replace('.mmd', 'bak.mmd')
    # print('save to ', newfilename)
    # ###覆盖源文件
    # with open(newfilename, 'w', encoding='utf-8') as output:
    #     output.write(newmmd_text)
    return newmmd_text



