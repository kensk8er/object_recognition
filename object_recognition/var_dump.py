# -*- coding: utf-8 -*-
"""
PHPのvar_dump的なもの
"""
from pprint import pprint
import types

def var_dump(obj):
	pprint(dump(obj))

def dump(obj):
	'''return a printable representation of an object for debugging'''
	newobj = obj
	if isinstance(obj, list):
		# リストの中身を表示できる形式にする
		newobj = []
		for item in obj:
			newobj.append(dump(item))
	elif isinstance(obj, tuple):
		# タプルの中身を表示できる形式にする
		temp = []
		for item in obj:
			temp.append(dump(item))
		newobj = tuple(temp)
	elif isinstance(obj, set):
	# セットの中身を表示できる形式にする
		temp = []
		for item in obj:
			# itemがclassの場合はdump()は辞書を返すが,辞書はsetで使用できないので文字列にする
			temp.append(str(dump(item)))
		newobj = set(temp)
	elif isinstance(obj, dict):
		# 辞書の中身（キー、値）を表示できる形式にする
		newobj = {}
		for key, value in obj.items():
			# keyがclassの場合はdump()はdictを返すが,dictはキーになれないので文字列にする
			newobj[str(dump(key))] = dump(value)
	elif isinstance(obj, types.FunctionType):
		# 関数を表示できる形式にする
		newobj = repr(obj)
	elif '__dict__' in dir(obj):
		# 新しい形式のクラス class Hoge(object)のインスタンスは__dict__を持っている
		newobj = obj.__dict__.copy()
		if ' object at ' in str(obj) and not '__type__' in newobj:
			newobj['__type__']=str(obj).replace(" object at ", " #").replace("__main__.", "")
		for attr in newobj:
			newobj[attr]=dump(newobj[attr])
	return newobj
