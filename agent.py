import streamlit as st
import pandas as pd
import numpy as np

# ───────────────────────────────────────────────
#   Columns that should NEVER be used for questioning
# ───────────────────────────────────────────────
DO_NOT_ASK_COLUMNS = {
    'id', 'code', 'code_prefix', 'product'
}

# ───────────────────────────────────────────────
#   Information Gain
# ───────────────────────────────────────────────
def calculate_info_gain(df):
    if len(df) <= 1:
        return {}
    
    n = len(df)
    total_entropy = np.log2(n) if n > 1 else 0.0
    
    gains = {}
    for col in df.columns:
        if col.lower() in DO_NOT_ASK_COLUMNS or col.lower().endswith(('_id', '_key', '_pk')):
            continue
        n_unique = df[col].nunique()
        if n_unique >= n * 0.85 or n_unique <= 1:
            continue
        
        grouped = df.groupby(col)
        weighted_entropy = sum(
            (len(sub) / n) * np.log2(len(sub)) if len(sub) > 1 else 0
            for _, sub in grouped
        )
        gain = total_entropy - weighted_entropy
        if gain > 1e-6:
            gains[col] = gain
    
    return gains

# ───────────────────────────────────────────────
#   Question generation
# ───────────────────────────────────────────────
def make_question(column, unique_values):
    uniques_sorted = sorted(set(unique_values))
    if len(uniques_sorted) <= 6:
        opts = ", ".join(map(str, uniques_sorted))
        return f"What is the **{column}**? (options: {opts})"
    else:
        sample = ", ".join(map(str, uniques_sorted[:4])) + ", ..."
        return f"What is the **{column}** of the item? (examples: {sample})"

# ───────────────────────────────────────────────
#   Load static test data (pandas only)
# ───────────────────────────────────────────────
@st.cache_data
def load_test_data():
    data = {
        'id': list(range(1, 25)),
        'code_prefix': ['A14']*8 + ['B27']*7 + ['C39']*5 + ['D42']*4,
        'product': [
            'Wireless Mouse', 'Wired Mouse', 'Gaming Mouse', 'Silent Office Mouse',
            'Vertical Ergonomic Mouse', 'Travel Compact Mouse', 'Ambidextrous Mouse', 'Bluetooth Mouse',
            'Mechanical Keyboard', 'Low-profile Keyboard', 'Gaming RGB Keyboard', 'Quiet Membrane Keyboard',
            'Ergonomic Split Keyboard', 'Wireless Compact Keyboard', 'Foldable Bluetooth Keyboard',
            '27" IPS Monitor', '32" 4K Monitor', '24" Office Monitor', 'Ultrawide 34" Curved',
            'Portable Monitor 15.6"', 'USB-C Dock', 'Thunderbolt 4 Dock', 'HDMI Switch 5-port', 'Multiport Adapter'
        ],
        'category': ['Mouse']*8 + ['Keyboard']*7 + ['Monitor']*5 + ['Dock/Adapter']*4,
        'brand': [
            'Logitech','Logitech','Razer','Logitech','Anker','Microsoft','Logitech','Logitech',
            'Keychron','Logitech','Razer','Logitech','Microsoft','Apple','Logitech',
            'Dell','LG','Dell','Samsung','ASUS','Anker','CalDigit','UGREEN','Satechi'
        ],
        'connectivity': [
            '2.4GHz','Wired USB','2.4GHz','Bluetooth','Bluetooth','2.4GHz','Wired','Bluetooth',
            'Wired USB','Bluetooth','Wired USB-C','Bluetooth','Bluetooth','Bluetooth','2.4GHz',
            'HDMI','HDMI + DP','HDMI','HDMI + DP + USB-C','USB-C',
            'USB-C','Thunderbolt','HDMI','USB-C + HDMI'
        ],
        'main_color': [
            'Black','Grey','Black RGB','White','Black','Grey','Black','White',
            'Black','White','Black RGB','Grey','Silver','Space Grey','Black',
            'Black','Silver','Black','Dark Grey','Silver',
            'Silver','Space Grey','Black','Silver'
        ],
        'price_range': [
            'Budget','Mid','Premium','Budget','Mid','Budget','Premium','Mid',
            'Mid','Budget','Premium','Mid','Premium','Premium','Mid',
            'Mid','Premium','Budget','Premium','Mid','Mid','Premium','Budget','Mid'
        ],
        'has_rgb': ['No']*8 + ['No','No','Yes','No','No','No','No', 'Yes'] + ['No']*8
    }
    df = pd.DataFrame(data)
    df['code'] = df['code_prefix'] + df['id'].astype(str).str.zfill(3)
    return df

# ───────────────────────────────────────────────
#   Apply all filters to the full dataset
# ───────────────────────────────────────────────
def apply_filters(full_df, filters):
    df = full_df.copy()
    for col, val in filters.items():
        if val:
            mask = df[col].astype(str).str.contains(val, case=False, na=False)
            df = df[mask]
    return df

# ───────────────────────────────────────────────
#   Streamlit App
# ───────────────────────────────────────────────
def main():
    st.title("Item Search Agent with Additive Filters")
    st.caption("Start with code, narrow down, then add/change filters to refine across all rows")

    # Initialize session state
    if 'all_items' not in st.session_state:
        st.session_state.all_items = load_test_data()
    
    if 'filters' not in st.session_state:
        st.session_state.filters = {'code': ''}  # dict of col: value for filters
    
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'phase' not in st.session_state:
        st.session_state.phase = "initial"
    
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None

    # Display chat
    for msg in st.session_state.chat_history:
        role = "You" if msg['role'] == 'user' else "Bot"
        st.markdown(f"**{role}:** {msg['content']}")

    # ── Current filters summary ─────────────────────
    st.markdown("**Current filters:** " + ", ".join(f"{k}={v}" for k,v in st.session_state.filters.items() if v))

    # ── Initial or update search ────────────────────
    if st.session_state.phase == "initial" or st.session_state.phase == "add_filter":
        code = st.text_input("Enter or update code fragment:", value=st.session_state.filters['code'], key="code_filter")
        if st.button("Apply code filter") and code.strip():
            st.session_state.filters['code'] = code
            st.session_state.current_df = apply_filters(st.session_state.all_items, st.session_state.filters)
            st.session_state.phase = "narrow"
            st.session_state.chat_history.append({"role": "bot", "content": f"Applied code filter '{code}'. Found **{len(st.session_state.current_df)}** items."})
            st.rerun()

    # ── Narrowing phase ─────────────────────────────
    if st.session_state.phase == "narrow":
        current = st.session_state.current_df
        
        if len(current) == 0:
            st.error("No items match current filters.")
            st.session_state.phase = "add_filter"
            st.rerun()
        
        st.markdown(f"**{len(current)} candidates:**")
        show_cols = ['id','code','product','category','brand','main_color','connectivity','price_range']
        st.dataframe(current[[c for c in show_cols if c in current.columns]])
        
        if len(current) == 1:
            st.success("Exact match found!")
            # Offer to add/change filters
            st.markdown("### Add or change a filter?")
            col = st.selectbox("Attribute", options=[c for c in st.session_state.all_items.columns if c not in DO_NOT_ASK_COLUMNS])
            val = st.text_input(f"Value for **{col}** (leave blank to remove filter)")
            if st.button("Apply filter change"):
                if val.strip():
                    st.session_state.filters[col] = val
                    msg = f"Added/updated filter: {col} = '{val}'"
                else:
                    st.session_state.filters.pop(col, None)
                    msg = f"Removed filter for {col}"
                st.session_state.chat_history.append({"role": "user", "content": msg})
                st.session_state.current_df = apply_filters(st.session_state.all_items, st.session_state.filters)
                st.session_state.chat_history.append({"role": "bot", "content": f"Re-searched with updated filters. Found **{len(st.session_state.current_df)}** items."})
                st.rerun()
            if st.button("Reset all filters"):
                st.session_state.filters = {'code': ''}
                st.session_state.phase = "initial"
                st.session_state.chat_history = []
                st.session_state.pending_question = None
                st.rerun()
            return
        
        # Ask disambiguation questions if needed
        if st.session_state.pending_question is None:
            gains = calculate_info_gain(current)
            if not gains:
                st.warning("No more ways to narrow — items too similar.")
                return
            best_col = max(gains, key=gains.get)
            uniques = current[best_col].dropna().unique()
            question = make_question(best_col, uniques)
            st.session_state.pending_question = {'column': best_col, 'text': question}
            st.session_state.chat_history.append({"role": "bot", "content": question})
            st.rerun()
        
        # Get answer
        answer = st.text_input("Your answer:", key="disamb_answer")
        if st.button("Submit"):
            if answer.strip():
                col = st.session_state.pending_question['column']
                st.session_state.filters[col] = answer  # Treat answer as new filter
                st.session_state.chat_history.append({"role": "user", "content": answer})
                st.session_state.current_df = apply_filters(st.session_state.all_items, st.session_state.filters)
                st.session_state.chat_history.append({"role": "bot", "content": f"Applied filter {col} = '{answer}'. Now **{len(st.session_state.current_df)}** items."})
                st.session_state.pending_question = None
                st.rerun()

if __name__ == "__main__":
    main()
