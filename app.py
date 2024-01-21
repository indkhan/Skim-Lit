from spacy.lang.en import English
import tensorflow as tf
import streamlit as st

st.title("Research Abstract Classifier")
st.markdown("""
        This app classifies study abstracts into different sections such as BACKGROUND, OBJECTIVE, METHODS, RESULTS, and CONCLUSIONS.
        Enter your study abstract in the text area below.
    """)
st.link_button("Reserch Paper National Library of Medicine", "https://pubmed.ncbi.nlm.nih.gov/")
abstext = st.text_area("Enter the study")

def split_chars(text):
        return " ".join(list(text))

loaded_model = tf.keras.models.load_model(
        r'models/nmodelmain')

if abstext:
    nlp = English()
    with st.spinner("Vectorizing..."):
        sentencizer = nlp.add_pipe("sentencizer")
        doc = nlp(abstext)
        abstract_lines = [str(sent) for sent in list(doc.sents)]

        total_lines_in_sample = len(abstract_lines)

        sample_lines = []
        for i, line in enumerate(abstract_lines):
            sample_dict = {}
            sample_dict["text"] = str(line)
            sample_dict["line_number"] = i
            sample_dict["total_lines"] = total_lines_in_sample - 1
            sample_lines.append(sample_dict)


        test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
        test_abstract_line_numbers_one_hot = tf.one_hot(
            test_abstract_line_numbers, depth=15)


        test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
        test_abstract_total_lines_one_hot = tf.one_hot(
            test_abstract_total_lines, depth=20)
        
        abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    with st.spinner("predicting......"):

        test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                    test_abstract_total_lines_one_hot,
                                                    tf.constant(abstract_lines),
                                                    tf.constant(abstract_chars)))


        test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)

    with st.spinner("Catogerizing...."):
        numlab = []

        for i in (test_abstract_preds):
            numlab.append(int(i))
        classnames = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']


        dictionary = {}
        for i in range(len(numlab)):
            class_name = classnames[numlab[i]]
            abstract_line = abstract_lines[i]
            if class_name in dictionary:
                dictionary[class_name].append(abstract_line)
            else:
                dictionary[class_name] = [abstract_line]
        ordered_dictionary = {}
        ordered_classnames = ['BACKGROUND', 'OBJECTIVE',
                            'METHODS', 'RESULTS', 'CONCLUSIONS']
        for class_name in ordered_classnames:
            if class_name in dictionary:
                ordered_dictionary[class_name] = dictionary[class_name]
        lis = []
        for class_name, lines in ordered_dictionary.items():
            lis.append(class_name)
            st.subheader(class_name)
            for line in lines:
                lis.append(line)
                st.write(line)
