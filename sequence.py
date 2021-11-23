import numpy as np
import pandas as pd

class Sequence(object):
    '''
    数列
    '''
    def __init__(self,*args,**kwargs):
        self._nvals_first_ = False
    
    def ival(self,i):
        '''
        第i个元素值
        '''
        raise(NotImplementedError())
        
    def nvals(self,n):
        '''
        前n个元素
        '''
        raise(NotImplementedError())
    
    def isum(self,i):
        '''
        前i个元素求和
        '''
        raise(NotImplementedError())
    
    def ipct(self,i):
        '''
        第i个元素占前i个元素和的比例
        '''
        a = self.ival(i)
        b = self.isum(i)
        return a/b
    
    def _format_idx_(self,i):
        if isinstance(i,slice):
            i = range(
                1 if i.start is None else i.start,
                i.stop+1,
                1 if i.step is None else i.step
            )
        return i

    def __getitem__(self,i):
        i = self._format_idx_(i)
        if self._nvals_first_:
            if isinstance(i,(list,range,tuple,set)):
                imax = max(i)
                items = self.nvals(imax)
                return [items[ii-1] if ii>0 else np.nan for ii in i ]
            return self.ival(i)
        else:
            i = self._format_idx_(i)
            if isinstance(i,(list,range,tuple,set)):
                return [self.ival(ii) for ii in i]
            return self.ival(i)


class ArithmeticSequence(Sequence):
    '''
    等差数列
    '''
    def __init__(self,a1,d):
        self.a1 = a1
        self.d = d
        self._nvals_first_ = False
        
    def ival(self,i):
        return self.a1+(i-1)*self.d
    
    def nvals(self,n):
        if n<1:
            return []
        
        res = [self.a1]
        for i in range(1,n):
            res.append(res[-1]+self.d)
        return res
    
    def isum(self,i):
        return self.a1*i+i*(i-1)*self.d/2.


class GeometricSequence(Sequence):
    '''
    等比数列
    '''
    def __init__(self,a1,q):
        self.a1 = a1
        self.q = q
        self._nvals_first_ = False
    
    def ival(self,i):
        if self.q ==1:
            an = self.a1
        else:
            an = self.a1*(self.q**(i-1))
        return an
    
    def nvals(self,n):
        if n<1:
            return []
        
        res = [self.a1]
        for i in range(1,n):
            res.append(res[-1]*self.q)
        return res
    
    def isum(self,i):
        if self.q==1:
            sn = i*self.a1
        else:
            sn = (self.a1-self.ival(i)*self.q)/(1-self.q)
        return sn


class Fibinacci(Sequence):
    '''
    菲波那契数列
    '''
    def __init__(self):
        self.golden = (np.sqrt(5)-1)/2
        self._nvals_first_ = True
    
    def ival(self,i):
        if i<1:
            return 0
        
        as_recursion = True if i>70 else False
        
        if as_recursion:
            a = 2*self.golden+1
            b = self.golden+1
            c = -self.golden
            return int(1/a*(b**i-c**i))
        else:
            if i==1:
                return 1
            if i==2:
                return 1
            a1 = a2 = 1
            for i in range(3,i+1):
                a1,a2 = a2,a1+a2
            return a2
    
    def nvals(self,n):
        items = [1,1]
        for j in range(3,n+1):
            items.append(items[-1]+items[-2])
        return items[:n]
    
    def isum(self,i):
        if i==1:
            return 1
        if i==2:
            return 2
        s = 2
        a1 = a2 = 1
        for i in range(3,i+1):
            a1,a2 = a2,a1+a2
            s = s+a2
        return s
    
class PrimeNumber(Sequence):
    '''
    ·质数只有两个因数：1和本身
    ·任何大于1的自然数，要么本身是质数，要么可以分解为几个质数之积，且这种分解是唯一的
    ·质数的个数是无限多的
    ·若n为正整数，在n²到(n+1)²之间至少有一个质数
    ·若n为大于等于2的正整数，则n到n!之间至少有一个质数
    ·若质数p为不超过n(n≥4)的最大质数，则p＞n/2
    ·所有大于10的质数中，个位数只有1，3，7，9
    ·若n是质数，则n可表达为6x+1或6x-1
    '''
    class PrimeNumberTheorem(object):
        def PNT(n):
            p = n/np.log(n)
            return p

        def PierreDusart(n):
            '''
            在n>346时生效
            '''
            p = PrimeNumber.PrimeNumberTheorem.PNT(n)
            bottom = np.floor(p * (1+0.992/np.log(n)))
            top = p*(1+1.2762/np.log(n))
            if isinstance(n,pd.Series):
                bottom[n<=346] = bottom[n<=346]-1
                bottom[n<=10] = 1
            else:
                bottom = bottom-1 if bottom<=346 else bottom
                bottom = 1 if bottom<=10 else bottom
            return np.floor(bottom),top

        def boundary(n):
            return PrimeNumber.PrimeNumberTheorem.PierreDusart(n)
        
        def number_for_n_prime_number(n):
            a = n
            b = n*n
            while a!=b:
                c = (a+b)/2
                nn = PrimeNumber.PrimeNumberTheorem.boundary(c)
                if nn[0]>n:
                    a,b = a,c
                elif nn[1]<n:
                    a,b = c,b
                else:
                    break
            return int(max(a,b)+1)
        
        def example(n = 1000000):
            vs = self.get_all_prime_numbers(n)
            info = pd.Series(0,range(2,n+1)).rename('is_prime')
            info.loc[vs] = 1
            info = info.to_frame()
            info['count'] = info['is_prime'].cumsum()
            info['bottom'],info['top'] = self.PrimeNumberTheorem.boundary(pd.Series(info.index,info.index))
            info['unvalid'] = ~np.logical_and(info['count']>=info['bottom'],info['count']<=info['top'])
            info.index.name = 'number'
            return info
    
    def __init__(self):
        self._nvals_first_ = True
    
    def sieve_of_eratosthenes(self,N):
        '''
        计算从0至N的素数
        '''
        N = N+1
        if N==1:
            return [0]
        
        is_prime = [1]*N
        # We know 0 and 1 are composites
        is_prime[0] = 0
        is_prime[1] = 0

        i = 2
        # This will loop from 2 to int(sqrt(x))
        while i*i <= N:
            # If we already crossed out this number, then continue
            if is_prime[i] == 0:
                i += 1
                continue

            j = 2*i
            while j < N:
                # Cross out this as it is composite
                is_prime[j] = 0
                # j is incremented by i, because we want to cover all multiples of i
                j += i
            i += 1
        return is_prime
    
    def get_all_prime_numbers(self,top,start=0):
        '''
        获取start到top里的所有素数
        '''
        is_prime = self.sieve_of_eratosthenes(top)[start:]
        return [i+start for i in range(1,len(is_prime)) if is_prime[i] == 1]
    
    def is_prime(self,n):
        # 0和1不是质数
        if n<=1:
            return False

        pre_primes = [2,3,5,7]
        for pp in pre_primes:
            if n==pp:
                return True

        for pp in pre_primes:
            if n%pp==0:
                return False

        sqrt = int(n**(1./2))+1
        for i in range(5,sqrt,6):
            if (n%i==0) or (n %(i+2)==0):
                return False
        return True
    
    def dispose(self,n):
        n = int(n)
        vs = self.get_all_prime_numbers(int(n**(1/2.)))
        res = []
        for v in vs:
            while n%v==0:
                res.append(v)
                n = n/v
        if len(res) == 0:
            res = [1,n]
        if n != 1:
            res.append(int(n))
        return res

    def ival(self,i):
        return self.nvals(i)[-1]
    
    def nvals(self,n):
        N = self.PrimeNumberTheorem.number_for_n_prime_number(n)
        return self.get_all_prime_numbers(N)[:n]
    
    def isum(self,i):
        return np.sum(self.nvals(i))
    
def example(n=10):
    for ss in [
    ArithmeticSequence(1,3),
    GeometricSequence(1,2),
    Fibinacci(),
    PrimeNumber()
    ]:
        print(ss.__class__.__name__,ss.nvals(n),ss[n],ss.isum(n),)

if __name__ == '__main__':
    example(4)
