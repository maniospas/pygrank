import setuptools
import sys, inspect
import pygrank.algorithms


def format(doc):
    if doc is None:
        return ""
    ret = ""
    prefix = ""
    open_example = False
    for line in doc.split("\n"):
        line = line.replace("\t", "   ").strip()
        if len(line) == 0:
            continue
        if open_example and ">>>" not in line:
            ret += "\n```\n"
            open_example = False
        line_break = "\n"
        if line == "Example:":
            line = "Example:\n\n```"
            prefix = ""
            line_break = "\n\n"
            open_example = True
        elif len(ret) == 0:
            line_break == ""
        elif len(prefix)>0 and ":" in line:
            line_break = "\n"+prefix
            line = "*"+line.replace(":",":*")
        elif len(prefix)>0 and ":" not in line:
            line_break = ""
        if line == "Attributes:":
            prefix = " * "
            line_break = "\n\n"
        ret += line_break+line+" "
    if open_example:
        ret += "\n```\n"
    return ret


from tests.example_graph import test_graph
G = test_graph()
def is_abstract(cls):
    try:
        cls().rank(G)
        return False
    except:
        return True


def generate_docs():
    text = "# :scroll: List of Graph Filters"
    text += "\nThe following filters can be imported from the package `pygrank.algorithms`. Constructor details" \
            "are provided. All of them can be used through the code patterns presented at the library's [documentation](documentation.md)." \
            " \n"

    text_abstract = ""
    for name, obj in inspect.getmembers(sys.modules["pygrank.algorithms"]):
        if inspect.isclass(obj) and issubclass(obj, pygrank.algorithms.abstract_filters.GraphFilter):
            abstract = is_abstract(obj)
            parents = [cls.__name__ for cls in inspect.getmro(obj)]
            extends = "" if len(parents)<=1 else "(["+parents[1]+"](#"+parents[1].lower()+"))"
            class_text = "\n### "+name+" "+extends+("\n *Abstract class*\n\n" if abstract else "")+"\n"+obj.__doc__
            for name, method in inspect.getmembers(obj):
                if name=="__init__":
                    class_text += " "+format(method.__doc__)
            if abstract:
                text_abstract += class_text
            else:
                text += class_text

    text += text_abstract

    with open("tutorials/graph_filters.md", "w") as file:
        file.write(text)


generate_docs()