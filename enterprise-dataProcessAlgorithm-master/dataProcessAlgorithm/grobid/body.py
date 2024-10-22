from xml.dom.minidom import parse
from xml.dom.minidom import parseString
def check_name(name_str):
    """
    :param name_str:标题名字
    :return: 是否保留，false不保留，true保留
    """
    # 标题为空
    if len(name_str) == 0:
        return False
    # 标题为致谢
    if name_str == 'Acknowledgements' or name_str == 'acknowledgements' and name_str == 'ACKNOWLEDGEMENTS':
        return False
    # 标题为纯数字或符号组成
    var = [i.isalpha() for i in name_str]
    rate = 1.0 - float(sum(var)) / len(name_str)
    if rate >= 0.99:
        return False
    return True


def block_process(blocks, print_str):
    # 处理每个div内容
    for block in blocks:
        for d_id in range(0, len(block.childNodes)):
            div_node = block.childNodes[d_id]
            if div_node.nodeName == "head":
                head_value = ''
                if div_node.attributes:
                    head_attrs = div_node.attributes.items()
                    for attr_id in range(0, len(head_attrs)):
                        if head_attrs[attr_id][0] == "n":
                            head_level = len(str(head_attrs[attr_id][1]).strip('.').split("."))
                            head_value += ('#' * max(head_level - 1, 0))
                if div_node.childNodes.length != 0:
                    head_name = str(div_node.childNodes[0].nodeValue)
                    if check_name(head_name):
                        head_value += ('##' + head_name + '\n')
                        print_str += head_value
            if div_node.nodeName == "p":
                print_str += '<p>\n'
                for p_id in range(0, len(div_node.childNodes)):
                    p_node = div_node.childNodes[p_id]
                    if p_node.nodeName == "s":
                        print_str += '<s coords="'
                        if p_node.attributes:
                            p_node_attrs = p_node.attributes.items()
                            for attr_id in range(0, len(p_node_attrs)):
                                if p_node_attrs[attr_id][0] == "coords":
                                    coords_value = str(p_node_attrs[attr_id][1])
                                    print_str += coords_value
                        print_str += '">'
                        for s_id in range(0, len(p_node.childNodes)):
                            s_node = p_node.childNodes[s_id]
                            if s_node.nodeName == "#text":
                                text_value = s_node.nodeValue
                                print_str += str(text_value)
                            if s_node.nodeName == "ref":
                                ref_str = '<ref'
                                if s_node.attributes:
                                    s_node_attrs = s_node.attributes.items()
                                    for attr_id in range(0, len(s_node_attrs)):
                                        ref_str += (' ' + str(s_node_attrs[attr_id][0]) + '="' + str(
                                            s_node_attrs[attr_id][1]) + '"')
                                ref_str += '>'
                                ref_value = ''
                                if s_node.childNodes.length > 0:
                                    ref_value = s_node.childNodes[0].nodeValue
                                    ref_str += str(ref_value)
                                    ref_str += "</ref>"
                                    print_str += ref_str
                        print_str += '</s>\n'
                print_str += '</p>\n'
            if div_node.nodeName == 'formula':
                print_str += '<formula coords="'
                if div_node.attributes:
                    p_node_attrs = div_node.attributes.items()
                    for attr_id in range(0, len(p_node_attrs)):
                        if p_node_attrs[attr_id][0] == "coords":
                            coords_value = str(p_node_attrs[attr_id][1])
                            print_str += coords_value
                print_str += '">'
                for f_id in range(0, len(div_node.childNodes)):
                    f_node = div_node.childNodes[f_id]
                    if f_node.nodeName == "#text":
                        text_value = f_node.nodeValue
                        print_str += str(text_value)
                    if f_node.nodeName == "label":
                        text_value = ''
                        if f_node.childNodes.length != 0:
                            text_value = f_node.childNodes[0].nodeValue
                        print_str += str(text_value)
                    if f_node.nodeName == "ref":
                        ref_str = '<ref'
                        if f_node.attributes:
                            f_node_attrs = f_node.attributes.items()
                            for attr_id in range(0, len(f_node_attrs)):
                                ref_str += (' ' + str(f_node_attrs[attr_id][0]) + '="' + str(
                                    f_node_attrs[attr_id][1]) + '"')
                        ref_str += '>'
                        ref_value = ''
                        if f_node.childNodes.length != 0:
                            ref_value = f_node.childNodes[0].nodeValue
                            ref_str += str(ref_value)
                            ref_str += "</ref>"
                            print_str += ref_str
                print_str += '</formula>\n'
    return print_str


def body_process(input_str, type):
    if len(input_str) == 0:
        return ''
    print_str = "#Main Body\n"
    # 通过字符串获取文件
    dom = parseString(input_str)
    # 获取文档元素对象
    elem = dom.documentElement
    # 获取 abstract和body中的div,存到[]blocks
    body = elem.getElementsByTagName('body')
    abstract = elem.getElementsByTagName('abstract')
    blocks = []
    if abstract:
        blocks = abstract[0].getElementsByTagName('div')
        if type == "paper":
            print_str += "[START_ABSTRACT]\n"
        print_str = block_process(blocks, print_str)
        if type == "paper":
            print_str += "[END_ABSTRACT]\n"
    if body:
        blocks = body[0].getElementsByTagName('div')
        print_str = block_process(blocks, print_str)
    out_str = print_str.replace("<p>\n</p>", "")
    return out_str


def content_process(input_str):
    if len(input_str) == 0:
        return ''
    out_str = "##Contents\n"
    # 通过字符串获取文件
    dom = parseString(input_str)
    # 获取文档元素对象
    elem = dom.documentElement
    body = elem.getElementsByTagName('body')
    blocks = []
    if body:
        blocks += body[0].getElementsByTagName('div')
    # 处理每个div内容
    for block in blocks:
        for d_id in range(0, len(block.childNodes)):
            div_node = block.childNodes[d_id]
            if div_node.nodeName == "head":
                head_level = ''
                if div_node.attributes:
                    head_attrs = div_node.attributes.items()
                    for attr_id in range(0, len(head_attrs)):
                        if head_attrs[attr_id][0] == "n":
                            head_level += (str(head_attrs[attr_id][1]).strip('.') + ' ')
                head_name = ""
                if div_node.childNodes.length > 0:
                    head_name = str(div_node.childNodes[0].nodeValue)
                    if check_name(head_name):
                        out_str += ('\t' * max(len(head_level.strip().split('.')) - 1,
                                               0) + '* ' + head_level + head_name + '\n')
    return out_str
