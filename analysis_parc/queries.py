

def crashes_with_lengths(df):

    return \
        df[ [ 'episode', 'crash' ] ] \
        .groupby( 'episode' ).filter( lambda rows: rows[['crash']].sum() > 0) \
        .groupby( 'episode' ).count() \
        .rename( columns={ 'crash':'length' } ) \
        .reset_index()


