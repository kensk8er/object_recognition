#!/usr/local/bin/python
from pprint import pprint
import types


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def pickle(data, file):
    import cPickle
    fo = open(file, 'w')
    cPickle.dump(data, fo, protocol=1)
    fo.close()


def var_dump(obj):
    pprint(dump(obj))


def dump(obj):
    '''return a printable representation of an object for debugging'''
    new_obj = obj
    if isinstance(obj, list):
        # リストの中身を表示できる形式にする
        new_obj = []
        for item in obj:
            new_obj.append(dump(item))
    elif isinstance(obj, tuple):
        # タプルの中身を表示できる形式にする
        temp = []
        for item in obj:
            temp.append(dump(item))
        new_obj = tuple(temp)
    elif isinstance(obj, set):
        # セットの中身を表示できる形式にする
        temp = []
        for item in obj:
            # itemがclassの場合はdump()は辞書を返すが,辞書はsetで使用できないので文字列にする
            temp.append(str(dump(item)))
        new_obj = set(temp)
    elif isinstance(obj, dict):
        # 辞書の中身（キー、値）を表示できる形式にする
        new_obj = {}
        for key, value in obj.items():
            # keyがclassの場合はdump()はdictを返すが,dictはキーになれないので文字列にする
            new_obj[str(dump(key))] = dump(value)
    elif isinstance(obj, types.FunctionType):
        # 関数を表示できる形式にする
        new_obj = repr(obj)
    elif '__dict__' in dir(obj):
        # 新しい形式のクラス class Hoge(object)のインスタンスは__dict__を持っている
        new_obj = obj.__dict__.copy()
        if ' object at ' in str(obj) and not '__type__' in new_obj:
            new_obj['__type__'] = str(obj).replace(" object at ", " #").replace("__main__.", "")
        for attr in new_obj:
            new_obj[attr] = dump(new_obj[attr])
    return new_obj
