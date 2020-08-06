import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
import itertools
import operator
import pandas as pd



class Bigmaxcut:
    def __init__(self,variables,numero,metodos ):
        self.variables = variables
        self.numero = numero
        self.metodos = metodos
        
    def fit(self,dataframe,target):
        self.dataframe = dataframe
        self.target = target
        self.champion_variable = {}
        self.champion_iv = {}
        for variable in self.variables:
            iv_dict_success = {}
            champion = {}
            print("variable: ",variable)
            for comb in list(itertools.product(self.metodos,list(range(2,self.numero+1)))):
                metodo=comb[0]
                numero=comb[1]
                if metodo!="arbol":
                    dataframe_ = self.dataframe[[variable,self.target]].copy()
                    cutter = KBinsDiscretizer(n_bins=numero, encode='ordinal', strategy=metodo)
                    try:
                        dataframe_[variable]=cutter.fit_transform(dataframe_)
                        edges = cutter.bin_edges_[0]
                        dict_edges = {}
                        for x in range(0,numero):
                            dict_edges.update({x:pd.Interval(left=edges[x], right=edges[x+1])})
                        dataframe_[variable]=dataframe_[variable].map(dict_edges)
                        dataframe_['n']=1
                        pivote=dataframe_[[variable,'n',self.target]].pivot_table(index=variable,columns=self.target,values='n',aggfunc='sum')
                        pivote.columns = ['0','1']
                        pivote['woe_'+variable] = np.log((pivote['0']/pivote['0'].sum())/(pivote['1']/pivote['1'].sum()))
                        pivote['iv_'+variable] = ((pivote['0']/pivote['0'].sum())-(pivote['1']/pivote['1'].sum()))*pivote['woe_'+variable]
                        iv=pivote['iv_'+variable].sum()
                        pivote = pivote[['woe_'+variable]].to_dict()['woe_'+variable]
                        champion.update({metodo+"_"+str(numero):iv})
                        iv_dict_success.update({metodo+"_"+str(numero):{"iv":iv,"woe_values":pivote}})
                    except:
                        champion.update({metodo+"_"+str(numero):0})
                        iv_dict_success.update({metodo+"_"+str(numero):{"iv":0,"woe_values":{}}})
                else:
                    dataframe_ = self.dataframe[[variable,self.target]].copy()
                    arbol = DecisionTreeClassifier(max_depth=numero)
                    arbol.fit(dataframe_[[variable]],dataframe_[self.target])
                    dataframe_['predict']=arbol.predict_proba(dataframe_[[variable]])[:,1] 
                    tabla = dataframe_[['predict',variable]].groupby(['predict']).agg([min,max])
                    tabla.columns = ["min","max"]
                    dataframe_["predict"] = dataframe_["predict"].map(dict(zip(list(tabla.index),list(map(lambda x,y: pd.Interval(left=x, right=y), tabla["min"], tabla["max"])))))
                    dataframe_['n']=1
                    pivote=dataframe_[["predict",'n',self.target]].pivot_table(index="predict",columns=self.target,values='n',aggfunc='sum')
                    pivote.columns = ['0','1']
                    pivote['woe_'+variable] = np.log((pivote['0']/pivote['0'].sum())/(pivote['1']/pivote['1'].sum()))
                    pivote['iv_'+variable] = ((pivote['0']/pivote['0'].sum())-(pivote['1']/pivote['1'].sum()))*pivote['woe_'+variable]
                    iv=pivote['iv_'+variable].sum()
                    pivote = pivote[['woe_'+variable]].to_dict()['woe_'+variable]
                    champion.update({"tree"+"_"+str(numero):iv})
                    iv_dict_success.update({"tree"+"_"+str(numero):{"iv":iv,"woe_values":pivote}})
            champion_ = sorted(champion.items(), key=operator.itemgetter(1),reverse=True)
            self.champion_variable.update({variable:iv_dict_success[champion_[0][0]]})
            self.champion_iv.update({variable:iv_dict_success[champion_[0][0]]["iv"]})
        return (self)
        
    def transform(self,dataframe):
        for variable in self.variables:
            dataframe[variable]=dataframe[variable].map(lambda x: next((v for k, v in self.champion_variable[variable]["woe_values"].items() if x in k),0))
        return(dataframe)