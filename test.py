class Autoencoder(nn.Module):
    
    
    def __init__(self):
        torch.manual_seed(0)
        super(Autoencoder,self).__init__()
        
        self.encode=nn.Sequential(nn.Linear(50,30),
                                  nn.LeakyReLU(),
                                  nn.Linear(30,20),
                                  nn.LeakyReLU(),
                                  nn.Linear(20,10),
                                  nn.Sigmoid())
        
        self.decode=nn.Sequential(nn.Linear(10,20),
                                  nn.LeakyReLU(),
                                  nn.Linear(20,30),
                                  nn.LeakyReLU(),
                                  nn.Linear(30,50),
                                  nn.LeakyReLU())
        self.net = nn.Sequential(nn.Linear(15,10),
                                 nn.Sigmoid(),
                                 nn.Linear(10,5),
                                 nn.Sigmoid())
        
        self.opt=optim.Adam(self.parameters(),lr=.3)
        
        
        
        
        
    def neuralNet(self,x1,x2):   # e.g x1-> (1000,50) and x2-> (1000,5)
        x1=self.encode(x1)  # (1000,50) -> (1000,10)
        print(x1.shape)
        x= torch.Tensor(np.concatenate((x1.detach().numpy(),x2.detach().numpy()),axis=1))  #(1000,15)
        print(x.shape)
        return self.net(x)
        
        
        
    def forward(self,x):
        x=self.encode(x)
        x=self.decode(x)
        return x
        
    
        

    def fit(self,x):
        
        for i in range(100):
            loss=[]
            accu_arr=[]
            
            y_hat=self.forward(x)
            ls=F.mse_loss(y_hat,x)
            print(ls)
            loss.append(ls.item())
            ls.backward()
            self.opt.step()
            self.opt.zero_grad()
            
        print("Final Loss", loss[-1])
        plt.plot(loss, 'r-')
        plt.show()
		
		


torch.manual_seed(0)
data, labels = make_blobs(n_samples=1000, centers=5, n_features=50, random_state=0)

sc = MinMaxScaler()
data = sc.fit_transform(data)


data_bin=(data>.5)*1
data_bin=torch.from_numpy(data_bin)

ae=Autoencoder()
ae.fit(data_bin.float())


            
            
            
