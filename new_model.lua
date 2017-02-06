require 'data'
require 'nn'
require 'cunn'
require 'rnn'

L1 = 'english'
L2 = 'turkish'
data_path = "./data/train_en_tr/"

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function test_model(final,epoch)
  local input1 = split:forward(inputs1[{{1,100},{}}])
  local input2 = split:forward(inputs2[{{1,100},{}}])
  local output1 = model1:forward( input1)
  local output2 = model2:forward( input2)
  all_roots1 = output1
  all_roots2 = output2

  score = 0
  for idx1 = 1, all_roots1:size(1) do
    closest = idx1
    for idx2 = 1, all_roots2:size(1) do
      if torch.dist(all_roots1[idx1],all_roots2[idx2]) < torch.dist(all_roots1[idx1],all_roots2[closest]) then
        closest = idx2
      end
    end
    
    if idx1 == closest then
      score = score + 1 
    else
      if final == true then print("Closest to: "..idx1.." is: ".. closest) end
    end
  end
  print("Test Score: " .. score.. '/' .. all_roots1:size(1))
  torch.save(L2..'_model'..epoch,model2)
  torch.save(L2..'_LT'..epoch,lt2)
  torch.save(L1..'_model'..epoch,model1)
  torch.save(L1..'_LT'..epoch,lt1)
end



lr = 0.5
emb_size = 128
lr_decay = 1.2
threshold = 3
max_epoch = 500
batch_size = 100
init = 0.01
max_seq_len = 100
win_size = 3
dump_freq = 50


corpus_en = Corpus(L1..".10000")
corpus_fr = Corpus(L2..".10000")
corpus_en:prepare_corpus(max_seq_len)
corpus_fr:prepare_corpus(max_seq_len)
no_of_sents = #corpus_en.sequences
inputs1=corpus_en:get_data():cuda()
inputs2=corpus_fr:get_data():cuda()

torch.save(L1..'_vocab',corpus_en.vocab_map)
torch.save(L2..'_vocab',corpus_fr.vocab_map)


vocab_size1 = corpus_en.no_of_words
vocab_size2 = corpus_fr.no_of_words

seq_len1 = corpus_en.longest_seq
seq_len2 = corpus_fr.longest_seq




-- Build the first model
model1 = nn.Sequential()
lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size) -- MaskZero
model1:add(nn.Sequencer(lt1))

mod1 = nn.ConcatTable()
for i = 1, seq_len1, 1 do -- seq length
  rec1 = nn.NarrowTable(i,win_size)
  mod1:add(rec1)
end

model1:add(mod1)
add_tan1 = nn:Sequential()
add_tan1:add(nn.Sequencer(nn.CAddTable()))
add_tan1:add(nn.Sequencer(nn.Tanh()))
add_tan1:add(nn.CAddTable()) -- add linear layer
model1:add(nn.MaskZero(add_tan1,1))
--nn.MaskZero(model1,1)
model1:getParameters():uniform(-1*init,init)

model1:cuda()

-- Build the second model
model2 = nn.Sequential()
lt2 = nn.LookupTableMaskZero(vocab_size2,emb_size) -- MaskZero
model2:add(nn.Sequencer(lt2))

mod2 = nn.ConcatTable()
for i = 1,  seq_len2, 1 do -- seq length
  rec2 = nn.NarrowTable(i,win_size)
  mod2:add(rec2)
end

model2:add(mod2)
add_tan2 = nn:Sequential()
add_tan2:add(nn.Sequencer(nn.CAddTable()))
add_tan2:add(nn.Sequencer(nn.Tanh()))
add_tan2:add(nn.CAddTable())
model2:add(nn.MaskZero(add_tan2,1))
--nn.MaskZero(model2,1)
model2:getParameters():uniform(-1*init,init)

model2:cuda()


criterion = nn.AbsCriterion():cuda()
nn.MaskZeroCriterion(criterion, 1)
beginning_time = torch.tic()
split = nn.SplitTable(2)
for i =1,max_epoch do
    errors = {}
    local inds = torch.range(1, no_of_sents,batch_size)
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    for j = 1, no_of_sents/batch_size do 
                --get input row and target
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+batch_size-1
        local input1 = split:forward(inputs1[{{start,endd},{}}])
        local input2 = split:forward(inputs2[{{start,endd},{}}])
        model1:zeroGradParameters()
        model2:zeroGradParameters()
        -- print( target)
        local output1 = model1:forward( input1)
        local output2 = model2:forward( input2)
        local err = criterion:forward( output1, output2)
        table.insert( errors, err)
        local gradOutputs = criterion:backward(output1, output2)
        model1:backward(input1, gradOutputs)
        model1:updateParameters(lr)
        output1 = model1:forward( input1)
        output2 = model2:forward( input2)
        err = criterion:forward( output2, output1)
        table.insert( errors, err)
        gradOutputs = criterion:backward(output2, output1)
        model2:backward(input2, gradOutputs)
        model2:updateParameters(lr)
    end
    printf ( 'epoch %4d, loss = %6.50f \n', i, torch.mean( torch.Tensor( errors))   )
    if i % threshold == 0 then lr = lr / lr_decay end
    if i % dump_freq == 0 then test_model(false,i) end
end

durationSeconds = torch.toc(beginning_time)
print('time elapsed:'.. printtime(durationSeconds))

test_model(true,'final')






--x = model1:forward(t)