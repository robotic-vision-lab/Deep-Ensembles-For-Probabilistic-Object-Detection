class Debugger:
    def __init__(self, debugBool = True):
        self.debug = debugBool
    def gIS(self):
        return 'g =  d.getDebugString'
    def getDebugString(self, expression):
        #takes a string to be evaluated by exec function
        return 'if d.gDS() == True: print(\'' + expression + ':\','+  expression+')'
    def T(self):
        return 'pdb.set_trace()'
    def gDS(self):
        return self.debug
    def sDS(self, db):
        if db is not None:
            self.debug = db
        else : self.debug = True if not self.debug else False


#d,e = Debugger(), exec
#g = d.getDebugString
#exec('if 0 == 0: print ("hi")')
#e(g('1+2'))
