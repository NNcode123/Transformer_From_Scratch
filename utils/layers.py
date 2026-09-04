from Model.transformer import Transformer
from pathlib import Path
from data.dataloader import TokenDataset, TokenLoader
import torch



def train(Model,optimizer_head, optim_dict_info, data, batches, epochs, start_epoch, file_dir_prefix = "Saved", is_distributed = False, mp = False):

    transformer: Transformer = Model()

    parent_dir = Path(__file__).resolve().parent

    if (not file_dir_prefix.startswith("Saved")): 
        raise RuntimeError("File_Prefix does not start with Saved. Please change it to do so")

    Path_obj  = parent_dir/Path(file_dir_prefix)

    Path_obj.mkdir(exists_ok = True)
    

    optimizer: torch.optim.Adam = optimizer_head(**optim_dict_info)

    data_loader = TokenLoader(data, batch_size=batches, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        tot_loss = 0
        for input_dict in data_loader:
            input_data = input_dict.get("src_ids")

            input_mask = input_dict.get("src_att_mask")

            output_data = input_dict.get("tgt_ids")

            output_mask = input_dict.get("tgt_mask")

            model_output = transformer.forward(input_data, output_data, input_mask,output_mask & torch.triu(torch.ones(output_mask.shape)) )

            loss = torch.nn.CrossEntropyLoss(model_output, output_data)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step

            tot_loss += loss.item()


        ap = tot_loss/len(data_loader)
        print(f"Epoch {epoch}| loss={ap:.4f}")

        file_path = Path_obj / f"epoch_{epoch}.pth"

        file_path.touch(exist_ok = True)

        torch.save({
            "epoch": epoch,
            "optim": optimizer.state_dict(),
            "model": transformer.state_dict(),
            "loss": ap
        }, file_path)

        


        







def translate_model(input_sentence_list, tokenizer ):
    pass








        



        















