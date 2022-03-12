论文：Retrieval Augmentation Reduces Hallucination in Conversation

项目：https://parl.ai/projects/hallucination/

In this work, we study the various components of retrieval-augmented neural architectures for dialogue – retrievers, rankers and encoder-decoders and propose several new variants, while analyzing which methods work well and in which situations they do so.

In particular, we improve downstream performance by employing Poly-encoder Transformers (Humeau et al., 2020) for fifiner-grained context-candidate scoring of documents, by proposing an iterative retrieval scheme where the retrieval improves through repetition, by employing end-to-end-trained retrievers in the Fusion-in-Decoder(Izacard and Grave, 2020) technique, and by building a dialogue turn-based retrieval mechanism that avoids the problem of standard retrievers that ignore much of the dialogue context.

RAG和FiD
- RAG Sequence和RAG Token

- FiD：the decoder can attend to all of the joint document/context representations at the same time when generating a response

Seq2Seq Models

- BART
- T5
- BlenderBot

Improving Retrieval

- Greater context-candidate interaction：context和condidate document之间有更多的interaction
	- Poly-encoders
	- ColBERT
- Iterative Retrieval
- Retriever-less Retrieval

Improving Augmented Generation
- Conditioning on Dialogue Turns
- Improving FiD







