require 'data_morph'
require 'nn'
require 'cunn'
require 'rnn'

L1 = 'turkish'
L2 = 'english'
data_path = "./data/train_en_tr/"

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function forward(word_model, sent_cvm, word_cvm, input)
  sent = nn.SplitTable(2):forward(input)
  words_emb = {}
  outputs = {}
  for w_i = 1, #sent do
    word = nn.SplitTable(2):forward(sent[w_i])
    --word_model:remember()
    parts_emb = word_model:forward(word)
    --parts_emb = {}
    --for p_i = 1, #word do
      --parts_emb[p_i] = word_model:forward(word[p_i])
    --end
    outputs[w_i] = parts_emb
    words_emb[w_i] = word_cvm:forward(parts_emb)
  end
  return sent_cvm:forward(words_emb)
  
end

function backward(word_model, sent_cvm, word_cvm, input, gradOutput)
  words_grads = sent_cvm:backward(words_emb,gradOutput)
  parts_grads = {}
  for w_i = #words_grads, 1, -1 do
    word = nn.SplitTable(2):forward(sent[w_i])
    parts_grads[w_i] = word_cvm:backward(outputs[w_i],words_grads[w_i])
    --for p_i = #word, 1, -1 do
      --word_model:backward(word[p_i],parts_grads[w_i][p_i])
    --end
    word_model:backward(word,parts_grads[w_i])
    --word_model:forget()
  end
  
 
  return '1'
  
end


function test_model(final,epoch)
  model_l1:evaluate()
  model_l2:evaluate()
  local input1 = inputs1[{{1,1000},{},{}}]
  local input2 = inputs2[{{1,1000},{},{}}]
  local output1 = model_l1:forward( input1)
  local output2 = model_l2:forward( input2)
  --local output1 = forward( morph_lt1, sent_cvm1,word_cvm1,input1)
  --local output2 = forward( morph_lt2, sent_cvm2,word_cvm2,input2)
  all_roots1 = output1:double()
  all_roots2 = output2:double()

  score = 0
  for idx1 = 1, all_roots1:size(1) do
    closest = 1
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
  torch.save(L2..'_model'..epoch,model_l2)
  torch.save(L2..'_LT'..epoch,lt2)
  torch.save(L1..'_model'..epoch,model_l1)
  torch.save(L1..'_LT'..epoch,lt1)
  model_l1:training()
  model_l2:training()
end

corpus_en = Corpus(L1..".1000.morph")
corpus_fr = Corpus(L2..".1000.tok")
corpus_en:prepare_corpus()
corpus_fr:prepare_corpus()
no_of_sents = #corpus_en.sequences
inputs1=corpus_en:get_data():cuda()
inputs2=corpus_fr:get_data():cuda()

torch.save(L1..'_vocab',corpus_en.vocab_map)
torch.save(L2..'_vocab',corpus_fr.vocab_map)


lr = 1
lr2 = 1
vocab_size1 = corpus_en.no_of_morphs
vocab_size2 = corpus_fr.no_of_morphs
emb_size = 64
lr_decay = 1.5
threshold = 1000
max_epoch = 600
batch_size = 50
dump_freq = 50
init = 10
init1 = 0.01

-- build model l1 
--model_l1 = nn.Sequential()
--local lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size)
--model_l1:add( nn.Sequencer(lt1))
--model_l1:add( nn.CAddTable())
--model_l1:getParameters():uniform(-0.01,0.01)

--model_l1:cuda()

morph_lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size)
--morph_lt1 = nn.Sequencer(morph_lt1,8):cuda()
--morph_lt1:remember()
-- morph_lt:getParameters():uniform(-0.01,0.01)
--morph_lt1:getParameters():uniform(-1*init,init)
word_cvm1 = nn.Sequential()
word_cvm1:add(nn.CAddTable())
word_cvm1:add(nn.Tanh())
--word_cvm1:add(nn.Linear(emb_size,emb_size))
word_cvm1:cuda()
--word_cvm1:getParameters():uniform(-1*init,init)
sent_cvm1 = nn.Sequential()
sent_cvm1:add(nn.CAddTable())
sent_cvm1:add(nn.Tanh())
--sent_cvm1:add(nn.Linear(emb_size,emb_size))
sent_cvm1:cuda()
--sent_cvm1:getParameters():uniform(-1*init,init)

local word_mod_add = nn.Sequential()
-- word_mod_add:add(nn.SplitTable(2))
word_mod_add:add(morph_lt1)
word_mod_add:add(nn.SplitTable(2))
lstm_inner = nn.LSTM(emb_size,emb_size)
word_mod_add:add(nn.Sequencer(lstm_inner)) -- this model outputs the word embeddings
word_mod_add:add(nn.SelectTable(-1))
--word_mod_add:add(nn.Sequencer(nn.CAddTable())) 
word_mod_add:add(nn.Tanh())
--word_mod_add:add(nn.Linear(emb_size,emb_size)) -- this model outputs the word embeddings
--word_mod_add:add(nn.Dropout()) -- this model outputs the word embeddings
--word_mod_add = nn.Sequencer(word_mod_add)
--word_mod_add:remember()
word_mod_add = nn.Recursor(word_mod_add)
model_l1 = nn.Sequential()
model_l1:add(nn.SplitTable(2))
model_l1:add(nn.Sequencer(word_mod_add))
lstm_outer = nn.LSTM(emb_size,emb_size)
model_l1:add(nn.Sequencer(lstm_outer)) -- this model outputs the sentence embeddings
model_l1:add(nn.SelectTable(-1)) -- this model outputs the sentence embeddings
model_l1:add(nn.Tanh()) -- this model outputs the sentence embeddings
-- model_l1:add(nn.Linear(emb_size,emb_size))
-- model_l1:add(nn.Dropout())
-- model_l1 = nn.Sequencer(model_l1)

--model_l1:getParameters():uniform(-1*init,init)
morph_lt1:getParameters():uniform(-1*init1,init1)
model_l1:cuda()



morph_lt2 = nn.LookupTableMaskZero(vocab_size2,emb_size)
morph_lt2 = nn.Sequencer(morph_lt2,8):cuda()
--morph_lt2:remember()
--morph_lt2:getParameters():uniform(-1*init,init)
word_cvm2 = nn.Sequential()
word_cvm2:add(nn.CAddTable())
word_cvm2:add(nn.Tanh())
--word_cvm2:add(nn.Linear(emb_size,emb_size))
word_cvm2:cuda()
--word_cvm2:getParameters():uniform(-1*init,init)
sent_cvm2 = nn.Sequential()
sent_cvm2:add(nn.CAddTable())
sent_cvm2:add(nn.Tanh())
--sent_cvm2:add(nn.Linear(emb_size,emb_size))
sent_cvm2:cuda()
--sent_cvm2:getParameters():uniform(-1*init,init)


local word_mod_add2 = nn.Sequential()
-- word_mod_add:add(nn.SplitTable(2))
word_mod_add2:add(morph_lt2)
word_mod_add2:add(nn.Sequencer(nn.SplitTable(2)))
--lstm = nn.LSTM(emb_size,emb_size)
--word_mod_add:add(nn.Sequencer(lstm)) -- this model outputs the word embeddings
--word_mod_add:add(nn.SelectTable(-1))
word_mod_add2:add(nn.Sequencer(nn.CAddTable())) 
word_mod_add2:add(nn.Sequencer(nn.Tanh()))
--word_mod_add:add(nn.Linear(emb_size,emb_size)) -- this model outputs the word embeddings
--word_mod_add:add(nn.Dropout()) -- this model outputs the word embeddings
--word_mod_add = nn.Sequencer(word_mod_add)
--word_mod_add:remember()

model_l2 = nn.Sequential()
model_l2:add(nn.SplitTable(2))
model_l2:add(word_mod_add2)
model_l2:add(nn.CAddTable()) -- this model outputs the sentence embeddings
model_l2:add(nn.Tanh()) -- this model outputs the sentence embeddings
-- model_l1:add(nn.Linear(emb_size,emb_size))
-- model_l1:add(nn.Dropout())
-- model_l1 = nn.Sequencer(model_l1)

--model_l2:getParameters():uniform(-1*init,init)
morph_lt2:getParameters():uniform(-1*init1,init1)
model_l2:cuda()



criterion = nn.AbsCriterion()
criterion = nn.MaskZeroCriterion(criterion, 1):cuda()
beginning_time = torch.tic()
split = nn.SplitTable(3)
model_l1:training()
model_l2:training()

for i =1,max_epoch do
    errors = {}
    local inds = torch.range(1, no_of_sents,batch_size)
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    for j = 1, no_of_sents/batch_size do 
                --get input row and target
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+batch_size-1
        local input1 = inputs1[{{start,endd},{},{}}]
        local input2 = inputs2[{{start,endd},{},{}}]
        model_l1:zeroGradParameters()
        model_l2:zeroGradParameters()
        morph_lt1:zeroGradParameters()
        sent_cvm1:zeroGradParameters()
        word_cvm1:zeroGradParameters()
        morph_lt2:zeroGradParameters()
        sent_cvm2:zeroGradParameters()
        word_cvm2:zeroGradParameters()
        -- print( target)
        --local output1 = forward(morph_lt1, sent_cvm1,word_cvm1,input1)
        --local output2 = forward(morph_lt2, sent_cvm2,word_cvm2,input2)
        local output1 = model_l1:forward( input1)
        local output2 = model_l2:forward( input2)
        local err = criterion:forward( output1, output2)
        table.insert( errors, err)
        local gradOutputs = criterion:backward(output1, output2)
        --backward(morph_lt1, sent_cvm1,word_cvm1,input1,gradOutputs)
        model_l1:backward(input1, gradOutputs)
        model_l1:updateParameters(lr)
        --morph_lt1:updateParameters(lr)
        --sent_cvm1:updateParameters(lr2)
        --word_cvm1:updateParameters(lr2)
        output1 = model_l1:forward( input1)
        --output1 = forward(morph_lt1, sent_cvm1,word_cvm1,input1)
        --output2 = forward(morph_lt2, sent_cvm2,word_cvm2,input2)
        output2 = model_l2:forward( input2)
        err = criterion:forward( output2, output1)
        table.insert( errors, err)
        gradOutputs = criterion:backward(output2, output1)
        model_l2:backward(input2, gradOutputs)
        --backward(morph_lt2, sent_cvm2,word_cvm2,input2,gradOutputs)
        model_l2:updateParameters(lr)
        morph_lt2:updateParameters(lr)
        sent_cvm2:updateParameters(lr2)
        word_cvm2:updateParameters(lr2)
    end
    printf ( 'epoch %4d, loss = %6.5f, lr = %1.4f \n', i, torch.mean( torch.Tensor( errors)),lr  )
    if i % threshold == 0 then lr = lr / lr_decay end
    if i % dump_freq == 0 then
      test_model(false,i)
    end
end

durationSeconds = torch.toc(beginning_time)
print('time elapsed:'.. printtime(durationSeconds))
test_model(true,'final')