require 'cunn'
require 'rnn'
local utils = require 'utils'
local manifold = require 'manifold'
local gfx = require 'gfx.js'

-- function to show an MNIST 2D group scatter plot:
local function show_scatter_plot(mapped_x, labels, h,w)

  -- count label sizes:
  local K = getn(labels)

  -- separate mapped data per label:
  mapped_data = {}
  for k = 1,K do
    mapped_data[k] = { key = labels[k], values = torch.Tensor(1, opts.ndims) }
  end
  local offset = torch.Tensor(K):fill(1)
  for n = 1,getn(labels) do
    mapped_data[n].values[offset[n]]:copy(mapped_x[n])
    offset[n] = offset[n] + 1
  end

  -- show results in scatter plot:
  local gfx = require 'gfx.js'
  gfx.chart(mapped_data, {
     chart = 'scatter',
     width = w,
     height = h,
  })
end


-- show map with original digits:
local function show_map(mapped_data, X)
  
  -- draw map with original digits:
  local im_size = 2048
  local background = 0
  local background_removal = true
  local map_im = manifold.draw_image_map(mapped_data, X:resize(X:size(1), 1, 28, 28), im_size, background, background_removal)
  
  -- plot results:
  local gfx = require 'gfx.js'
  gfx.image(map_im)
end


-- function that performs demo of t-SNE code on MNIST:
local function demo_tsne()


  -- load data 
  local lt1 = torch.load('exp_all_82_add/english_LT'):double()
  local lt2 = torch.load('exp_all_82_add/turkish_LT'):double()
  local vocab_en = torch.load('exp_all_82_add/english_vocab')
  local vocab_tr = torch.load('exp_all_82_add/turkish_vocab')
  -- words_count1 = table.getn(vocab_en)
  -- words_count2 = table.getn(vocab_tr)
  embeddings_en = lt1:getParameters()
  embeddings_tr = lt2:getParameters()
  embeddings_en:resize(getn(vocab_en),embeddings_en:size()[1]/getn(vocab_en))
  embeddings_tr:resize(getn(vocab_tr),embeddings_tr:size()[1]/getn(vocab_tr))
  torch.save('embeddings_en',embeddings_en)
  torch.save('embeddings_tr',embeddings_tr)
  
  lbl_en = get_labels(vocab_en)
  lbl_tr = get_labels(vocab_tr)
  torch.save('labels_en',lbl_en)
  torch.save('labels_tr',lbl_tr)
  embeddings_en = embeddings_en:double()
  embeddings_tr = embeddings_tr:double()
  
  
  -- prepare data 
  -- get from excel sheet
  -- words = {{'sweet','candy'},{'center','middle'},{'change','transform'},{'conflict','clash'},{'disappear','vanish'}} -- get from excel sheet
  no_of_words = table.getn(words)*2
  x = torch.DoubleTensor(no_of_words,embeddings_en:size(2))
  labels = {}
  idx = 1
  for _,word in ipairs(words) do 
    table.insert(labels,word[1])
    table.insert(labels,word[2])
    en = vocab_en[word[1]]+1 -- add 1 to skip the padding
    tr = vocab_tr[word[2]]+1
    x[idx] = embeddings_en[en]
    x[idx+1] = embeddings_en[tr]
    idx = idx + 2 
  end
  
  
  -- run t-SNE:
  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 5}
  mapped_x1 = manifold.embedding.tsne(x, opts)
  print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
  --gfx.chart(mapped_x1, {chart='scatter'})
  show_scatter_plot(mapped_x1, labels, 300)
  --show_map(mapped_x1, x:clone())

end


-- function that performs demo of t-SNE code on MNIST:
local function demo_tsne_sents()


  -- load data 
  local model_en = torch.load('exp_all_82_add/english_model'):double()
  local model_tr = torch.load('exp_all_82_add/turkish_model'):double()
  
  local lt1 = torch.load('exp_all_82_add/english_LT')
  local lt2 = torch.load('exp_all_82_add/turkish_LT')

  local vocab_en = torch.load('exp_all_82_add/english_vocab')
  local vocab_tr = torch.load('exp_all_82_add/turkish_vocab')
  -- words_count1 = table.getn(vocab_en)
  -- words_count2 = table.getn(vocab_tr)
  embeddings_en = lt1:getParameters()
  embeddings_tr = lt2:getParameters()
  embeddings_en:resize(getn(vocab_en),embeddings_en:size()[1]/getn(vocab_en))
  embeddings_tr:resize(getn(vocab_tr),embeddings_tr:size()[1]/getn(vocab_tr))
  torch.save('embeddings_en',embeddings_en)
  torch.save('embeddings_tr',embeddings_tr)
  
  lbl_en = get_labels(vocab_en)
  lbl_tr = get_labels(vocab_tr)
  torch.save('labels_en',lbl_en)
  torch.save('labels_tr',lbl_tr)
  embeddings_en = embeddings_en:double()
  embeddings_tr = embeddings_tr:double()
  
  
  -- prepare data 
  -- sents = {{'cat','kedi'},{'horse','at'},{'cow','inek'},{'dog','köpek'}} -- get from excel sheet
  -- sents = {{'one','bir'},{'two','iki'},{'three','üç'},{'four','dört'},{'five','beş'}}
   sents = {{[[Hi. Today, I'm going to take you through glimpses of about eight of my projects, done in collaboration with Danish artist Soren Pors. We call ourselves Pors and Rao, and we live and work in India.]],[[Merhaba, Bugün, seni Danimarkalı artist Soren Pors ile yaptığım sekiz projeye göz atman için alacağım. Biz birbirimize Pors ve Rao diye sesleniriz, ve biz hindistanda hem çalışıyoruz hem de oturuyoruz.]]},
       {[[And if you want to summarize in one just single word, well, this is what we're trying to do. We're trying to give a future to our past in order to have a future. As long as we live a life of curiosity and passion, there is a bit of Leonardo in all of us. Thank you.]],[[Ve eğer sadece bir kelime ile özetlenmek istersen, güzel, bu bizim yapmaya çalıştığımız şeydir. Geleceğe sahip olmak için biz geçmişimize bir gelecek vermek için çalışıyoruz. Meraklı ve tutkulu bir yaşam yaşadığımız sürece, hepimizin içinde bir parça Leonardo olacak. Teşekürler.]]},
       {[[My subject today is learning. And in that spirit, I want to spring on you all a pop quiz. Ready? When does learning begin? Now as you ponder that question, maybe you're thinking about the first day of preschool or kindergarten, the first time that kids are in a classroom with a teacher. Or maybe you've called to mind the toddler phase when children are learning how to walk and talk and use a fork. Maybe you've encountered the Zero-to-Three movement, which asserts that the most important years for learning are the earliest ones. And so your answer to my question would be: Learning begins at birth.]],[[Benim bugünkü konum öğrenme. Ve bu ruh haliyle, size ani bir quiz yapacağım. Hazır mısınız? Öğrenme ne zaman başlıyor? Şimdi siz bu soru hakkında düşünürken, belki okul öncesi veya anaokuldaki ilk gününüzü düşünüyorsunuz, ilk defa öğrencilerin sınıfta öğretmenlenle birlikte olduğu zaman.]]}}  -- get from excel sheet
  -- sents = {{' And so this is five-foot, five-inches of frost that she left behind.',' Ve bu beş adımlık kırağıyı.. ...arkasında bırakıyor. '},{"How are you?",'Nasılsınız?'},{'Thanks.','Teşekkürler.'}} -- get from excel sheet
  
  no_of_sents = table.getn(sents)*2
  x = torch.DoubleTensor(no_of_sents,embeddings_en:size(2))
  labels = {}
  idx = 1
  for _,sent in ipairs(sents) do 
    table.insert(labels,string.sub(sent[1],1,50)..".....")
    table.insert(labels,string.sub(sent[2],1,50)..".....")
    
    -- get idxs
    en = {}
    tr = {}
    words1 = stringx.split(sent[1])
    words2 = stringx.split(sent[2])
    for _,w in ipairs(words1) do
      if vocab_en[w] == nil then 
        w = '<pad>' 
      end
      table.insert(en, torch.DoubleTensor(1):fill(vocab_en[w]))
    end
    
    for _,w in ipairs(words2) do
      print(w)
      if vocab_tr[w] == nil then 
        w = '<pad>' 
      end
      table.insert(tr, torch.DoubleTensor(1):fill(vocab_tr[w]))
    end
    
    root_en = model_en:forward(en)
    root_tr = model_tr:forward(tr)
    
    x[idx] = root_en
    x[idx+1] = root_tr
    idx = idx + 2 
  end
  
  
  -- run t-SNE:
  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 2}
  mapped_x1 = manifold.embedding.tsne(x, opts)
  print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
  --gfx.chart(mapped_x1, {chart='scatter'})
  show_scatter_plot(mapped_x1, labels, 500,500)
  --show_map(mapped_x1, x:clone())

end


local function padding(sequence,max_len)
    -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , max_len - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (max_len - math.min(max_len,#sequence))+1, max_len do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end


local function demo_tsne_sents_new(max_len)


  -- load data 
  local model_en = torch.load('exp_all_tok_add_68/english_model'):double()
  local model_tr = torch.load('exp_all_tok_add_68/turkish_model'):double()
  
  local lt1 = torch.load('exp_all_tok_add_68/english_LT')
  local lt2 = torch.load('exp_all_tok_add_68/turkish_LT')

  local vocab_en = torch.load('exp_all_tok_add_68/english_vocab')
  local vocab_tr = torch.load('exp_all_tok_add_68/turkish_vocab')
  -- words_count1 = table.getn(vocab_en)
  -- words_count2 = table.getn(vocab_tr)
  embeddings_en = lt1:getParameters()
  embeddings_tr = lt2:getParameters()
  embeddings_en:resize(getn(vocab_en),embeddings_en:size()[1]/getn(vocab_en))
  embeddings_tr:resize(getn(vocab_tr),embeddings_tr:size()[1]/getn(vocab_tr))
  torch.save('embeddings_en',embeddings_en)
  torch.save('embeddings_tr',embeddings_tr)
  
  lbl_en = get_labels(vocab_en)
  lbl_tr = get_labels(vocab_tr)
  torch.save('labels_en',lbl_en)
  torch.save('labels_tr',lbl_tr)
  embeddings_en = embeddings_en:double()
  embeddings_tr = embeddings_tr:double()
  
  
  -- prepare data 
  --sents = {{'cat','kedi'},{'horse','at'},{'cow','inek'},{'dog','köpek'}} -- get from excel sheet
  -- sents = {{'one','bir'},{'two','iki'},{'three','üç'},{'four','dört'},{'five','beş'}}
  sents = {{[[Hi. Today, I'm going to take you through glimpses of about eight of my projects, done in collaboration with Danish artist Soren Pors. We call ourselves Pors and Rao, and we live and work in India.]],[[Merhaba, Bugün, seni Danimarkalı artist Soren Pors ile yaptığım sekiz projeye göz atman için alacağım. Biz birbirimize Pors ve Rao diye sesleniriz, ve biz hindistanda hem çalışıyoruz hem de oturuyoruz.]]},
       {[[And if you want to summarize in one just single word, well, this is what we're trying to do. We're trying to give a future to our past in order to have a future. As long as we live a life of curiosity and passion, there is a bit of Leonardo in all of us. Thank you.]],[[Ve eğer sadece bir kelime ile özetlenmek istersen, güzel, bu bizim yapmaya çalıştığımız şeydir. Geleceğe sahip olmak için biz geçmişimize bir gelecek vermek için çalışıyoruz. Meraklı ve tutkulu bir yaşam yaşadığımız sürece, hepimizin içinde bir parça Leonardo olacak. Teşekürler.]]},
       {[[My subject today is learning. And in that spirit, I want to spring on you all a pop quiz. Ready? When does learning begin? Now as you ponder that question, maybe you're thinking about the first day of preschool or kindergarten, the first time that kids are in a classroom with a teacher. Or maybe you've called to mind the toddler phase when children are learning how to walk and talk and use a fork. Maybe you've encountered the Zero-to-Three movement, which asserts that the most important years for learning are the earliest ones. And so your answer to my question would be: Learning begins at birth.]],[[Benim bugünkü konum öğrenme. Ve bu ruh haliyle, size ani bir quiz yapacağım. Hazır mısınız? Öğrenme ne zaman başlıyor? Şimdi siz bu soru hakkında düşünürken, belki okul öncesi veya anaokuldaki ilk gününüzü düşünüyorsunuz, ilk defa öğrencilerin sınıfta öğretmenlenle birlikte olduğu zaman.]]}}
  -- sents = {{'six','altı'},{'seven','yedi'},{'eight','sekiz'},{'nine','dokuz'},{'ten','on'}}
   -- sents = {{'And	so	this	is	five-foot	,	five-inches	of	frost	that	she	left	behind	.','Ve	bu	beş	adımlık	kırağıyı  ,	arkasında	bırakıyor	.'},{"My	name	is",'Benim	adım'},{'Me	and	my	sister','Ben	ve	kız	kardeşim'}} -- get from excel sheet
    
  no_of_sents = table.getn(sents)*2
  x = torch.DoubleTensor(no_of_sents,embeddings_en:size(2))
  labels = {}
  idx = 1
  for _,sent in ipairs(sents) do 
    table.insert(labels,string.sub(sent[1],1,50)..".....")
    table.insert(labels,string.sub(sent[2],1,50)..".....")
    
    -- get idxs
    en = {}
    tr = {}
    words1 = stringx.split(sent[1],' ')
    words2 = stringx.split(sent[2],' ')
    
    words1 = padding(words1,100)
    words2 = padding(words2,100)
    
    for _,w in ipairs(words1) do
      if vocab_en[w] == nil then 
        w = '<pad>' 
      end
      table.insert(en, torch.DoubleTensor(1):fill(vocab_en[w]))
    end
    
    for _,w in ipairs(words2) do
      -- print(w)
      if vocab_tr[w] == nil then 
        w = '<pad>' 
      end
      table.insert(tr, torch.DoubleTensor(1):fill(vocab_tr[w]))
    end
    
    root_en = model_en:forward(en)
    root_tr = model_tr:forward(tr)
    
    x[idx] = root_en
    x[idx+1] = root_tr
    idx = idx + 2 
  end
  
  
  -- run t-SNE:
  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 2}
  mapped_x1 = manifold.embedding.tsne(x, opts)
  for i = 1, mapped_x1:size(1),2 do
     closest = i+1
    for j = 2, mapped_x1:size(1),2 do
      if torch.dist(mapped_x1[i],mapped_x1[j]) < torch.dist(mapped_x1[i],mapped_x1[closest]) then
        closest = j
      end
    end
    print("Closest to: "..i.." is: ".. closest)
  end
  print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
  --gfx.chart(mapped_x1, {chart='scatter'})
  show_scatter_plot(mapped_x1, labels, 500,500)
  --show_map(mapped_x1, x:clone())

end


local function demo_tsne_sents_new_bow(max_len)


  -- load data 
  local model_en = torch.load('exp_all_tok_tan_40/english_model'):double()
  local model_tr = torch.load('exp_all_tok_tan_40/turkish_model'):double()
  
  local lt1 = torch.load('exp_all_tok_tan_40/english_LT'):double()
  local lt2 = torch.load('exp_all_tok_tan_40/turkish_LT'):double()

  local vocab_en = torch.load('exp_all_tok_tan_40/english_vocab')
  local vocab_tr = torch.load('exp_all_tok_tan_40/turkish_vocab')
  -- words_count1 = table.getn(vocab_en)
  -- words_count2 = table.getn(vocab_tr)
  embeddings_en = lt1:getParameters()
  embeddings_tr = lt2:getParameters()
  embeddings_en:resize(getn(vocab_en),embeddings_en:size()[1]/getn(vocab_en))
  embeddings_tr:resize(getn(vocab_tr),embeddings_tr:size()[1]/getn(vocab_tr))
  torch.save('embeddings_en',embeddings_en)
  torch.save('embeddings_tr',embeddings_tr)
  
  lbl_en = get_labels(vocab_en)
  lbl_tr = get_labels(vocab_tr)
  torch.save('labels_en',lbl_en)
  torch.save('labels_tr',lbl_tr)
  embeddings_en = embeddings_en:double()
  embeddings_tr = embeddings_tr:double()
  
  
  -- prepare data 
  -- sents = {{'cat','kedi'},{'horse','at'},{'cow','inek'},{'dog','köpek'}} -- get from excel sheet
  -- sents = {{'one','bir'},{'two','iki'},{'three','üç'},{'four','dört'},{'five','beş'}}
  -- sents = {{'six','altı'},{'seven','yedi'},{'eight','sekiz'},{'nine','dokuz'},{'ten','on'}}
    sents = {{'And	so	this	is	five-foot	,	five-inches	of	frost	that	she	left	behind	.','Ve	bu	beş	adımlık	kırağıyı..	...	arkasında	bırakıyor	.'},{"My	name	is",'Benim	adım'},{'Me	and	my	sister','Ben	ve	kız	kardeşim'}} -- get from excel sheet
    
  
  no_of_sents = table.getn(sents)*2
  x = torch.DoubleTensor(no_of_sents,embeddings_en:size(2))
  labels = {}
  idx = 1
  for _,sent in ipairs(sents) do 
    table.insert(labels,string.sub(sent[1],1,50)..".....")
    table.insert(labels,string.sub(sent[2],1,50)..".....")
    
    -- get idxs
    en = {}
    tr = {}
    words1 = stringx.split(sent[1], '\t')
    words2 = stringx.split(sent[2], '\t')
    
    --words1 = padding(words1,100)
    --words2 = padding(words2,100)
    
    for _,w in ipairs(words1) do
      if vocab_en[w] == nil then 
        w = '<pad>' 
      end
      table.insert(en, torch.DoubleTensor(1):fill(vocab_en[w]))
    end
    
    for _,w in ipairs(words2) do
      -- print(w)
      if vocab_tr[w] == nil then 
        w = '<pad>' 
      end
      table.insert(tr, torch.DoubleTensor(1):fill(vocab_tr[w]))
    end
    
    root_en = nn.CAddTable():forward(nn.Sequencer(lt1):forward(en))[1]
    root_tr = nn.CAddTable():forward(nn.Sequencer(lt2):forward(tr))[1]
    
    x[idx] = root_en
    x[idx+1] = root_tr
    idx = idx + 2 
  end
  
  
  -- run t-SNE:
  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 2}
  mapped_x1 = manifold.embedding.tsne(x, opts)
  for i = 1, mapped_x1:size(1),2 do
     closest = i+1
    for j = 2, mapped_x1:size(1),2 do
      if torch.dist(mapped_x1[i],mapped_x1[j]) < torch.dist(mapped_x1[i],mapped_x1[closest]) then
        closest = j
      end
    end
    print("Closest to: "..i.." is: ".. closest)
  end
  print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
  --gfx.chart(mapped_x1, {chart='scatter'})
  show_scatter_plot(mapped_x1, labels, 500,500)
  --show_map(mapped_x1, x:clone())

end


-- run the demo:
demo_tsne_sents_new(99)


