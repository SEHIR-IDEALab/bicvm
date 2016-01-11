require 'data'
require 'nn'
require 'cunn'
require 'rnn'

function printf(s,...)
  return io.write(s:format(...))
end


corpus_en = Corpus("fr-en.en.10")
corpus_fr = Corpus("fr-en.fr.10")
corpus_en:prepare_corpus()
corpus_fr:prepare_corpus()
no_of_sents = #corpus_en.sequences
inputs1=corpus_en:get_data()
inputs2=corpus_fr:get_data()

lr = 0.01
vocab_size1 = corpus_en.no_of_words
vocab_size2 = corpus_fr.no_of_words
emb_size = 64
lr_decay = 2.5
threshold = 10
max_epoch = 20
batch_size = 2

additive1 = nn.Sequential()
lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size)
additive1:add( nn.Sequencer(lt1))
additive1:add( nn.CAddTable())
additive1:getParameters():uniform(-0.01,0.01)

additive1:cuda()

additive2 = nn.Sequential()
lt2 = nn.LookupTableMaskZero(vocab_size2,emb_size)
additive2:add( nn.Sequencer(lt2))
additive2:add( nn.CAddTable())
additive2:getParameters():uniform(-0.01,0.01)

additive2:cuda()



criterion = nn.AbsCriterion():cuda()
nn.MaskZeroCriterion(criterion, 1)
beginning_time = torch.tic()
for i =1,max_epoch do
    errors = {}
    local shuffle = torch.totable(torch.randperm(inputs1:size(1)))
    for j = 1, no_of_sents,batch_size do 
                --get input row and target
        local split = nn.SplitTable(2)
        local input1 = split:forward(inputs1[{{j,j+batch_size-1},{}}])
        local input2 = split:forward(inputs2[{{j,j+batch_size-1},{}}])
        additive1:zeroGradParameters()
        additive2:zeroGradParameters()
        -- print( target)
        local output1 = additive1:forward( input1)
        local output2 = additive2:forward( input2)
        local err = criterion:forward( output1, output2)
        table.insert( errors, err)
        local gradOutputs = criterion:backward(output1, output2)
        additive1:backward(input1, gradOutputs)
        additive1:updateParameters(lr)
        output1 = additive1:forward( input1)
        output2 = additive2:forward( input2)
        err = criterion:forward( output2, output1)
        table.insert( errors, err)
        gradOutputs = criterion:backward(output2, output1)
        additive2:backward(input2, gradOutputs)
        additive2:updateParameters(lr)
    end
    printf ( 'epoch %4d, loss = %6.8f \n', i, torch.mean( torch.Tensor( errors))   )
    if i % threshold == 0 then lr = lr / lr_decay end
end

all_roots1 = {}
all_roots2 = {}

for j = 1,no_of_sents do 
                --get input row and target
        local input1 = inputs1[j]:split(1)
        local input2 = inputs2[j]:split(1)
        local output1 = additive1:forward( input1)
        local output2 = additive2:forward( input2)
        all_roots1[j] = torch.CudaTensor(emb_size)
        all_roots2[j] = torch.CudaTensor(emb_size)
        nn.rnn.recursiveCopy(all_roots1[j],output1)
        nn.rnn.recursiveCopy(all_roots2[j],output2)
end
score = 0
for idx1 = 1, #all_roots1 do
    closest = 1
    for idx2 = 2, #all_roots2 do
      if torch.dist(all_roots1[idx1],all_roots2[idx2]) < torch.dist(all_roots1[idx1],all_roots2[closest]) then
        closest = idx2
      end
    end
    print("Closest to: "..idx1.." is: ".. closest)
    if idx1 == closest then
      score = score + 1 
    end
end
print("Test Score: " .. score.. '/' .. table.getn(all_roots1))
