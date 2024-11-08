def create_data_loader(
    torch_dataset: data.Dataset,
    batch_size,
    shuffle,
    sampler=None,
    num_workers=0,
    np_collate=False,
    max_squared_res=1e6,
    length_batch=False,
    drop_last=False,
    prefetch_factor=2,
    num_gpus=1,
):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    elif length_batch:
        if num_gpus > 1:
            collate_fn = lambda x: possible_tuple_length_batching_multi_gpu(
                x, max_squared_res=max_squared_res, num_gpus=num_gpus
            )
        else:
            collate_fn = lambda x: possible_tuple_length_batching(
                x, max_squared_res=max_squared_res
            )
    else:
        collate_fn = None

    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=True,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context="fork"
        if num_workers != 0
        else None,  # TODO Try without. Doesn't seem to matter
    )