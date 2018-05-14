from random import sample

def init_lstm(lstm, hidden_size, T_max):
    for name, params in lstm.state_dict().items():
        if "weight" in name:
            nn.init.xavier_uniform_(params)
        elif "bias" in name:
            init = torch.log(torch.rand(hidden_size)*(T_max - 1) + 1)
            params[:hidden_size] = -init.clone()
            params[hidden_size:2*hidden_size] = init

    return lstm

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FastaiWrapper():
    def __init__(self, model, crit):
        self.model = model
        self.crit = crit
    
    def get_layer_groups(self, precompute=False):
        return self.model

def create_fake_data(self, num_batches=20):
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

        num_samples = 64*num_batches
        trn_X = np.zeros(shape=(num_samples, self.CH, self.R, self.C))
        trn_y = np.zeros(shape=(num_samples, 1))
        val_X = np.zeros(shape=(num_samples//6, self.CH, self.R, self.C))
        val_y = np.zeros(shape=(num_samples//6, 1))
        trn = [trn_X, trn_y]
        val = [val_X, val_y]
        fake_data = ImageClassifierData.from_arrays("./data", trn=trn, val=val,
                                        classes=classes)
        return fake_data

def forward(self, X):
        pass 

def train_controller(self, _=None, __=None):
    batch = sample(self.memories, self.batch_size)

    search_probas = []
    policies = []
    values = []
    scores = []

    value_loss = 0

    search_probas = []
    values = []
    scores = []

    for i, memory in enumerate(batch):
        trajectory = memory["trajectory"]
        score = memory["score"]
        decision_idx = memory["decision_idx"]
        
        cont_out, hidden = self.controller(self.first_emb.view(1, 1, -1))

        for emb_idx in trajectory:
            emb = self.embeddings[emb_idx].view(1, 1, -1)
            cont_out, hidden = self.controller(emb, hidden)
            value = self.value_head(cont_out).view(-1)
            values.append(value)
            scores.append(score)
        logits = self.softmaxs[decision_idx](cont_out).squeeze()
        probas = F.softmax(logits.unsqueeze(0), dim=1).squeeze()
        policies.append(probas)
        search_probas.append(memory["search_probas"])
            
    scores = torch.tensor(scores).float()
    if len(values) > 0:
        values = torch.cat(values).float()
    else:
        assert len(trajectory) == 0

    if self.has_cuda:
        scores = scores.cuda()
        if len(values) > 0: values = values.cuda()

    if len(values) > 0: value_loss = F.mse_loss(values, scores)

    search_probas = torch.cat(search_probas)
    policies = torch.cat(policies)

    if self.has_cuda:
        search_probas = search_probas.cuda()
        policies = policies.cuda()

    search_probas_loss = -search_probas.unsqueeze(0).mm(torch.log(policies.unsqueeze(-1)))
    search_probas_loss /= self.batch_size

    total_loss = search_probas_loss 
    if len(values) > 0:
        total_loss += value_loss

    return total_loss

def fastai_train(self, controller, memories, batch_size, num_cycles=10, epochs=1, min_memories=None):
    self.memories = memories
    self.batch_size = batch_size
    if min_memories is None:
        min_memories = batch_size*30

    if (len(memories) < min_memories):
        print("Have {} memories, need {}".format(len(memories), min_memories))
        return
    controller_wrapped = FastaiWrapper(model=controller, crit=self.train_controller)
    controller_learner = Learner(data=self.fake_data, models=controller_wrapped)
    controller_learner.crit = controller_wrapped.crit
    controller_learner.opt_fn = optim.Adam
    controller_learner.model.train()

    controller_learner.model.real_forward = controller_learner.model.forward

    controller_learner.model.forward = lambda x: x
    controller_learner.fit(8e-3, epochs, wds=1e-7) #was 7e-2
    # controller_learner.fit(2, epochs, cycle_len=num_cycles, use_clr_beta=(10, 13.68, 0.95, 0.85), 
    #     wds=1e-4)

    controller_learner.model.forward = controller_learner.model.real_forward

    del self.memories
    del self.batch_size
    del controller_learner.model.real_forward
    del controller_learner

def controller_lr_find(self, controller, memories, batch_size, start_lr=1e-5, end_lr=2):
    self.memories = memories
    self.batch_size = batch_size
    if (len(memories) < batch_size):
        print("Have {} memories, need {}".format(len(memories), batch_size))
        return
    arch = FastaiWrapper(model=controller, crit=self.train_controller)

    arch.model.real_forward = arch.model.forward
    arch.model.forward = lambda x: x

    learn = Learner(data=self.fake_data, models=arch)
    learn.opt_fn = optim.Adam
    learn.crit = arch.crit
    learn.model.train()

    learn.lr_find(start_lr=start_lr, end_lr=end_lr)
    learn.model.forward = learn.model.real_forward

    return learn