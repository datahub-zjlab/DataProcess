##coding:utf-8
from grobid.grobid_parser.parse import _string_to_tree,_parse_biblio

xml_ns = "http://www.w3.org/XML/1998/namespace"
ns = "http://www.tei-c.org/ns/1.0"

####输入xml文件，返回refereces列表
def get_references(filename):
    xml_text = open(filename, "r", encoding='utf-8').read()
    tree = _string_to_tree(xml_text)
    tei = tree.getroot()
    header = tei.find(f".//{{{ns}}}teiHeader")
    if header is None:
        raise ValueError("XML does not look like TEI format")

    refs = []
    for (i, bs) in enumerate(tei.findall(f".//{{{ns}}}listBibl/{{{ns}}}biblStruct")):
        ref = _parse_biblio(bs)
        ref.index = i
        author_names = []
        t_level = 'J'
        title = ref.title
        for author in ref.authors:
            author_names.append(author.full_name)

        ###参考文献类型，书籍，论文，网站等
        if ref.series_title is not None:
            t_level = 'A'
            title = ref.series_title
        elif ref.url:
            # print('check url', ref.url)
            t_level = 'OWL'

        citation = str(i+1)+'.[' + str(i+1) + ']' + '(<' + ref.id + '>'
        if len(ref.authors) == 0:  ###无作者
            if ref.institution:
                citation += ref.institution
            elif ref.journal:
                citation += ref.journal
            elif ref.editors:
                citation += ref.editors
            citation += '.' + str(title) + '.' + '[' + t_level + ']' + ','###题目和类型
        else:
            citation += ','.join(author_names) + '.' + str(title) + '.' + '[' + t_level + ']' + ',' + str(ref.journal)

        if ref.publisher:
            citation += str(ref.publisher)
        elif ref.url:
            citation += str(ref.url)
        if ref.volume:
             citation += ',vol:' + str(ref.volume)
        if ref.pages:
            citation += ',pages:' + str(ref.pages)
        citation += ',' + str(ref.date)+'.)'
        # print('check ref', citation)
        refs.append(citation)
    return '\n'.join(refs)


if __name__ == '__main__':
    filename = '21-%283-4%29-3662.tei.xml'
    get_references(filename)
    # revise_citations(filename)
    # print(docdict)
