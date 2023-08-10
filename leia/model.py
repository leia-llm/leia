from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class LeiaLlamaConfig(LlamaConfig):
    def __init__(
        self,
        entity_vocab_size: int = 1000000,
        similarity_function: str = "cosine",
        temperature: float = 1.0,
        layer_index: int = 31,
        use_entity_decoder_activation: bool = False,
        use_entity_prediction: bool = True,
        use_entity_prev_token_prediction: bool = True,
        use_entity_last_token_prediction: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.similarity_function = similarity_function
        self.temperature = temperature
        self.layer_index = layer_index
        self.use_entity_decoder_activation = use_entity_decoder_activation
        self.use_entity_prediction = use_entity_prediction
        self.use_entity_prev_token_prediction = use_entity_prev_token_prediction
        self.use_entity_last_token_prediction = use_entity_last_token_prediction


@dataclass
class LeiaCausalLMOutputWithPast(CausalLMOutputWithPast):
    lm_loss: torch.FloatTensor | None = None
    entity_prev_token_loss: torch.FloatTensor | None = None
    entity_prev_token_accuracy: torch.FloatTensor | None = None
    entity_last_token_loss: torch.FloatTensor | None = None
    entity_last_token_accuracy: torch.FloatTensor | None = None


class LeiaEntityDecoder(nn.Module):
    def __init__(self, config: LeiaLlamaConfig):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.entity_vocab_size, bias=False)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        token_logits = self.decoder(token_embeddings)
        return token_logits


class LeiaEntityPredictionHead(nn.Module):
    def __init__(self, config: LeiaLlamaConfig):
        super().__init__()
        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = LeiaEntityDecoder(config)

    def forward(self, hidden_states: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # shape: [batch, entity, hidden]
        token_embeddings = torch.gather(
            hidden_states, 1, token_positions.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        # shape: [batch, entity, hidden]
        token_embeddings = self.dense(token_embeddings)

        if self.config.similarity_function == "cosine":
            token_embeddings = F.normalize(token_embeddings, dim=-1)
        else:
            assert (
                self.config.similarity_function == "dot"
            ), f"Invalid similarity function: {self.config.similarity_function}"

        if self.config.use_entity_decoder_activation:
            token_embeddings = ACT2FN[self.config.hidden_act](token_embeddings)

        logits = self.decoder(token_embeddings)

        if self.config.temperature != 1.0:
            logits /= self.config.temperature

        return logits


class LeiaLlamaForCausalLM(LlamaForCausalLM):
    _no_split_modules = ["LlamaDecoderLayer", "LeiaEntityDecoder"]
    _keys_to_ignore_on_load_missing = [r"last_token_head", r"prev_token_head"]

    def __init__(self, config: LeiaLlamaConfig):
        super().__init__(config)
        if config.use_entity_prev_token_prediction:
            self.prev_token_head = LeiaEntityPredictionHead(config)
        if config.use_entity_last_token_prediction:
            self.last_token_head = LeiaEntityPredictionHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
        entity_prev_token_positions: torch.Tensor | None = None,
        entity_last_token_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> LeiaCausalLMOutputWithPast:
        kwargs["return_dict"] = True
        if entity_ids is not None:
            kwargs["output_hidden_states"] = True

        result = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        result = LeiaCausalLMOutputWithPast(
            loss=result.loss,
            logits=result.logits,
            past_key_values=result.past_key_values,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
            lm_loss=result.loss.detach().clone() if result.loss is not None else None,
        )
        if self.config.use_entity_prediction and self.training:
            for prefix, head, token_positions in [
                ("entity_prev_token", self.prev_token_head, entity_prev_token_positions),
                ("entity_last_token", self.last_token_head, entity_last_token_positions),
            ]:
                if not self.config.use_entity_prev_token_prediction and prefix == "entity_prev_token":
                    continue
                if not self.config.use_entity_last_token_prediction and prefix == "entity_last_token":
                    continue

                no_entity = token_positions.sum() == 0
                token_logits = head(result.hidden_states[self.config.layer_index], token_positions)
                if no_entity:
                    # prevent loss from being NaN
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

                entity_prediction_loss = loss_fn(
                    token_logits.view(-1, self.config.entity_vocab_size), entity_ids.view(-1)
                )
                entity_prediction_accuracy = (
                    (token_logits.argmax(-1) == entity_ids).float().masked_select(entity_ids != 0).mean()
                )

                if no_entity:
                    result.loss += 0.0 * entity_prediction_loss
                else:
                    result.loss += 1.0 * entity_prediction_loss
                    result[f"{prefix}_loss"] = entity_prediction_loss.detach()
                    result[f"{prefix}_accuracy"] = entity_prediction_accuracy

        return result
