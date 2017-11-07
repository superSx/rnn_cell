from tensorflow.contrib import rnn
import tensorflow as tf
class DMN():
    def __init__(self,q,s,d0,r):
        self.n_hidden=165
        self.question=tf.placeholder(dtype=tf.float32,shape=[None,d0,q])
        self.answer=tf.placeholder(dtype=tf.float32,shape=[None,d0,s])
        def bilstm(name,inputs,n_hidden,length,init_state_fw=None,init_state_bw=None):
            with tf.variable_scope('inner'+name) as scope:
                inputs=tf.unstack(inputs,num=length,axis=2)
                lstm_fw=rnn.LSTMBlockCell(n_hidden)
                lstm_bw=rnn.LSTMBlockCell(n_hidden)
                outputs,fw,bw=rnn.static_bidirectional_rnn(lstm_fw,lstm_bw,inputs,dtype=tf.float32,initial_state_fw=
                                                           init_state_fw,initial_state_bw=init_state_bw)
            return tf.transpose(outputs,[1,2,0]),tf.tanh(tf.concat(fw,bw,axis=1))
        ques,q_out=bilstm('',self.question,self.n_hidden,length=q)
        ans,a_out=bilstm('',self.answer,self.n_hidden,q_out,q_out)



        seq=tf.concat(self.question,self.answer,axis=2)
        full_seq,context=bilstm('seq',seq,self.n_hidden,length=s)
class GRUCell():
    def __init__(self,input_size,n_hidden,question,q,type='fw',modified=False):
        self.n_hidden=n_hidden
        self.input_size=input_size
        self.question=question
        self.q=q
        self.Wz=tf.get_variable('Wz',shape=[self.input_size,self.n_hidden])
        self.Wr=tf.get_variable('Wr',shape=[self.n_hidden,self.n_hidden])
        def normal_gru(x,h,input_size,hidden_size):
            with tf.variable_scope('gru'):
                Wz=tf.get_variable('Wz',shape=[input_size,hidden_size])
                Wr=tf.get_variable('Wr',shape=[input_size,hidden_size])
                W=tf.get_variable('W',shape=[input_size,hidden_size])
                Uz=tf.get_variable('Uz',shape=[hidden_size,hidden_size])
                Ur=tf.get_variable('Ur',shape=[hidden_size,hidden_size])
                U=tf.get_variable('U',shape=[hidden_size,hidden_size])
                bz=tf.get_variable('bz',shape=[hidden_size])
                br=tf.get_variable('br',shape=[hidden_size])
                bh=tf.get_variable('bh',shape=[hidden_size])
                print(x,h)
                z=tf.sigmoid(tf.matmul(x,Wz)+tf.matmul(h,Uz)+bz)
                r=tf.sigmoid(tf.matmul(x,Wr)+tf.matmul(h,Ur)+br)
                h_hat=tf.tanh(tf.matmul(x,W)+tf.matmul(r*h,U)+bh)
                h=z*h+(1-z)*h_hat
            return h
        def modified_gru(x,h,input_size,hidden_size):
            with tf.variable_scope('modified_gru'):
                z=g(x,h,self.n_hidden)
                h=z*normal_gru(x,h,input_size=input_size,hidden_size=hidden_size)+(1-z)*h
            return h
        def z(x,y):
            return tf.sqrt(tf.square(x-y))
        def g(c,m,n_hidden):
            with tf.variable_scope('g'):
                W1=tf.get_variable('w1',shape=[n_hidden,n_hidden])
                W2=tf.get_variable('w2',shape=[n_hidden,n_hidden])
                b1=tf.get_variable('b1',shape=[n_hidden])
                b2=tf.get_variable('b2',shape=[n_hidden])
                G=tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(z(c,m),W1)+b1),W2)+b2)
            return G
        def modified_gru_step(h,x):
            e=modified_gru(x,h,self.n_hidden,self.n_hidden)
            return e
        def normal_gru_step(h,x):
            m=normal_gru(x,h,self.input_size,self.n_hidden)
            return m
        def caculate(elements):
            es,e=[],self.q
            # elements=tf.unstack(elements,num=20,axis=2)
            # initial_state=tf.zeros([200],name='initial_state')
            # print(tf.concat(values=(elements,elements),axis=2))
            if modified:
                es=tf.scan(modified_gru_step,elems=elements,initializer=e)
                e=es[-1]
            else:
                es=tf.scan(normal_gru_step,elems=elements,initializer=e)
                e=es[-1]
            return es,e
        def bi_caculate(elements):
            es,es_fw,es_bw,e1,e2=[],[],[],self.q,self.q
            print(elements)
            if modified:
                with tf.variable_scope('modified_fw'):
                    es_fw=tf.scan(modified_gru_step,elems=elements,initializer=e1)
                with tf.variable_scope('modified_bw'):
                    es_bw=tf.reverse(tf.scan(modified_gru_step,elems=tf.reverse(elements,axis=[0]),initializer=e1),axis=[0])
                e=tf.concat(values=(es_fw[-1],es_bw[-1]),axis=1)
                es=tf.concat(values=(es_fw,es_bw),axis=2)
            else:
                with tf.variable_scope('fw'):
                    es_fw=tf.scan(normal_gru_step,elems=elements,initializer=e2)
                with tf.variable_scope('bw'):
                    es_bw=tf.reverse(tf.scan(normal_gru_step,elems=tf.reverse(elements,axis=[0]),initializer=e2),axis=[0])
                e=tf.concat(values=(es_fw[-1],es_bw[-1]),axis=1)
                es=tf.concat(values=(es_fw,es_bw),axis=2)
            return es,e
        if type=='bi':
            self.output,self.e=bi_caculate(self.question)
        if type=='fw':
            self.output,self.e=caculate(self.question)

if __name__=='__main__':
    with tf.variable_scope('111') as scope:
        tf.get_variable('1',shape=[1])

    with tf.variable_scope('111',reuse=True):
        tf.get_variable('1',shape=[1])
    x1=tf.placeholder(dtype=tf.float32,shape=[20,None,100])

    hidden=tf.matmul(x1[:,0,:], tf.zeros([100, 200]))
    gru=GRUCell(100,200,x1,hidden,type='bi')
    print(tf.transpose(gru.output,[1,0,2]))
